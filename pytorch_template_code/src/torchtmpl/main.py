# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib
import subprocess 
import datetime 
from functools import partial
from collections import defaultdict


# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
from torchvision import transforms
import torch.nn.functional as F
import tqdm
import time
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform 
import PIL 
from PIL import Image 
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
import torch.nn.functional as F_nn

# Local imports
from . import data
from . import models
from . import optim
from . import utils

NUM_CLASSES = 86

def train_sweep(tmp_testpath=None, tmp_trainpath=None):
    """
    This has to be called with 
    """
    print("Nouvelle run")
    print("Nouvelle run, nouveau code, gros gain")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    print(f"We are using {device} for training")

    num_classes = NUM_CLASSES

    with wandb.init() as run:
        config = run.config
        config = utils.unflatten_config(config) #We unflatten the dicitonary given to wandb with this utility function  
        wandb_log = wandb.log
        logging.info(config)
        
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
        

        train_config = config["train"]

        

        # Build the model
        logging.info("= Model")
        model_config = config["model"]
        model_name = model_config["class"]
        train_transform = None
        valid_transform = None 

        if "pretrained_path" in model_config:
            pretrained_path = model_config["pretrained_path"]
            logging.info(f"Instantiating {model_name} with pre-trained weights.")
            
            
            model_class = getattr(models.pretrained_models, model_name)
            model = model_class(
                pretrained_path=pretrained_path,
                pretrained = True, 
                num_classes=num_classes, 
            )

            # 2. Conditionally freeze the backbone using the unified interface
            freeze_pretrained = model_config.get("freeze_pretrained", False)
            if freeze_pretrained:
                logging.info("Freezing the pre-trained backbone.")
                for param in model.get_backbone().parameters():
                    param.requires_grad = False
                
                # Ensure the classifier block remains strictly unfrozen
                for param in model.get_classifier().parameters():
                    param.requires_grad = True
            else:
                logging.info("Fine-tuning the entire model (backbone unfrozen).")


            if "old_model_path" in model_config:
                old_model_path = model_config["old_model_path"]
                logging.info(f"Loading model state from {old_model_path}")
                model.load_state_dict(torch.load(old_model_path, weights_only=True), strict=False)

            
            
            data_cfg = resolve_data_config(model.pretrained_cfg, model=model.get_backbone())
            train_transform = create_transform(**data_cfg, is_training=True, scale=(0.8, 1.0), color_jitter=0,hflip=0)
            valid_transform = create_transform(**data_cfg, is_training=False)
        
        pretrained_in_color = model_config.get("pretrained_in_color", False)#To know if the pretrained_model takes Black and White pictures as inputs or RGB images

        # Build the dataloaders
        logging.info("= Building the dataloaders")
        data_config = config["data"]
        batch_size = data_config["batch_size"]

        train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
            data_config, use_cuda, train_transform=train_transform,valid_transform=valid_transform,tmp_trainpath=tmp_trainpath,pretrained_in_color=pretrained_in_color 
        )

        base_targets = train_loader.dataset.subset.dataset.targets
        class_counts = torch.bincount(torch.tensor(base_targets)) #for the focal loss and the weighted loss

        if not "pretrained_path" in model_config:
            logging.info("We are not using a pretrained model ie custom model !")
            model = models.build_model(model_config, input_size, num_classes)
            if "old_model_path" in model_config:
                old_model_path = model_config["old_model_path"]
                logging.info(f"Loading model at {old_model_path}")
                model.load_state_dict(torch.load(model_config["old_model_path"],weights_only=True))
        
        model.to(device)

        

        # Build the loss
        logging.info("= Loss")
        loss_config = config["loss"]
        is_weighted = loss_config["is_weighted"]
        lossname = loss_config["lossname"]
        normalized_name = lossname.strip().lower().replace("_", "")
        print(f"We are using the loss : {normalized_name}")
        focal_loss_set = ("FocalLoss", "focalloss","Focalloss", "focalLoss", "focal_loss", "FocalLoss ")
        if normalized_name in focal_loss_set:
            logging.info("We are using a Focal Loss")
            loss = optim.get_focal_loss(class_counts, device, loss_config["gamma"])
        elif is_weighted:
            is_article_weighted = loss_config["is_article_weighted"]
            loss = optim.get_weighted_loss(lossname, class_counts, device, is_article_weighted=is_article_weighted)
            logging.info("We are using a weighted loss")
        else:
            loss = optim.get_loss(loss_config, config["data"]["trainpath"], device)
            logging.info("We are using a regular (non weighted) loss")

        # Build the optimizer
        logging.info("= Optimizer")
        optim_config = config["optim"]
        optimizer = optim.get_optimizer(optim_config, filter(lambda p: p.requires_grad, model.parameters()))
        
        logging.info("= Scheduler")
        scheduler = optim.get_scheduler(optimizer, config)
        

        logging.info(f"We are running the latest code ! Yay !")
        # Build the callbacks
        logging_config = config["logging"]
        # The logname is the pretrained path if it exists, the name of the base model if it doesn't
        if "pretrained_path" in model_config and model_config["pretrained_path"]:
            logname = model_config["pretrained_path"].replace("/", "_").replace(":", "_")
        else:
            logname = model_config["class"]

        raw_logdir = os.path.expandvars(logging_config["logdir"])
        save_dir = os.path.expandvars(logging_config["save_dir"]) #To instantiate the ${JOB_WORKSPACE} temp variable

        logdir = utils.generate_unique_logpath(raw_logdir, logname)
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
            logging.info(f"created a logdir at {logdir}")

        logging.info(f"Will be logging into {os.path.abspath(logdir)}")

        # Copy the config file into the logdir
        logdir = pathlib.Path(logdir)
        
        with open(logdir / "config.yaml", "w") as file:
            ###### ADD THE NECESSARY STUFF TO THE TEST CONFIG FILE FOR EASIER TESTING !!!!
            yaml.dump(config, file)
            file.write(f"test:\n    model_path: [{os.path.abspath(logdir)}/best_model.pt,{os.path.abspath(logdir)}/best_model_loss.pt]\n    save_dir: {save_dir}")

        # Make a summary script of the experiment
        input_size = next(iter(train_loader))[0].shape
        summary_text = (
            f"Logdir : {logdir}\n"
            + "## Command \n"
            + " ".join(sys.argv)
            + "\n\n"
            + f" Config : {config} \n\n"
            + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
            + "## Summary of the model architecture\n"
            + f"{torchinfo.summary(model, input_size=input_size)}\n\n"
            + "## Loss\n\n"
            + f"{loss}\n\n"
            + "## Datasets : \n"
            + f"Train : {train_loader.dataset.subset.dataset}\n"
            + f"Validation : {valid_loader.dataset.subset.dataset}"
        )
        with open(logdir / "summary.txt", "w") as f:
            f.write(summary_text)
        logging.info(summary_text)
        if wandb_log is not None:
            wandb.log({"summary": summary_text})
        
        # Define the early stopping callback
        model_checkpoint_f1score = utils.ModelCheckpoint(
            model, str(logdir / "best_model.pt"), min_is_best=False
        )
        
        # Define the early stopping callback
        model_checkpoint_loss = utils.ModelCheckpoint(
            model, str(logdir / "best_model_loss.pt"), min_is_best=True
        )

        # Early stopping callback to save the model every 50 epochs even if the test loss is not bettering 
        model_checkpoint_50_epochs =  utils.ModelCheckpoint(
            model, str(logdir / "last_model.pt"), min_is_best=True
        )

        is_dynamic = sys.stdout.isatty()
        if is_dynamic:
            logging.info("We are running in a dynamic environment (the tqdm bar will be shown)")
        else:
            logging.info("We are not running in an interactive environment so to speed up training, the tqdm bar will not be shown")

        
        for e in range(train_config["nepochs"]):
            logging.info("Entering a new epoch")
            # Train 1 epoch
            time_before_training = time.time()
            train_loss = utils.train(model, train_loader, loss, optimizer, device,dynamic_display=is_dynamic,batch_size = batch_size)
            time_of_training = (time.time() - time_before_training )/60
            logging.info(f"This epoch took {time_of_training} minutes to train")

            time_before_test = time.time()
            test_loss, f1score = utils.evaluate(model, valid_loader, loss, num_classes, device)
            time_of_test = (time.time() - time_before_test) / 60 
            logging.info(f"Validation (loss + f1score) took {time_of_test:.2f} minutes")


            updated_loss = model_checkpoint_loss.update(test_loss)
            logging.info(
                "[%d/%d] Test loss : %.3f %s"
                % (
                    e,
                    train_config["nepochs"],
                    test_loss,
                    "[>> BETTER LOSS <<]" if updated_loss else "",
                )
            )

            updated_score = model_checkpoint_f1score.update(f1score)
            logging.info(
                "[%d/%d] F1 SCORE : %.3f %s"
                % (
                    e,
                    train_config["nepochs"],
                    f1score,
                    "[>> BETTER F1SCORE<<]" if updated_score else "",
                )
            )

            if e % 50 == 0:
                #calling with -e ensures that the model is saved since -e is strictly decreasing
                epoch_update = model_checkpoint_50_epochs.update(-e)
                logging.info(
                "[%d/%d] Test loss : %.3f %s"
                % (
                    e,
                    train_config["nepochs"],
                    test_loss,
                    "[>> LATEST <<]" if epoch_update else "",
                )
                )

            if updated_loss:
                logging.info(f"We are logging an artifact due to loss improvement !")
                artifact = wandb.Artifact(name="best-model-loss",type ="model",metadata={"loss" : test_loss, "epoch" : e})
                artifact.add_file(model_checkpoint_loss.savepath)
                wandb.log_artifact(artifact)
            
            if updated_score:
                logging.info(f"We are logging an artifact due to f1score improvement!")
                artifact = wandb.Artifact(name="best-model-f1score",type ="model",metadata={"score" : f1score, "epoch" : e})
                artifact.add_file(model_checkpoint_f1score.savepath)
                wandb.log_artifact(artifact)

            #Update the learning rate
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            logging.info(f"Epoch {e} complete. New LR: {current_lr:.6f}")

            # Update the dashboard
            metrics = {"train_CE": train_loss, "test_CE": test_loss, "f1score": f1score}
            if wandb_log is not None:
                logging.info("Logging on wandb")
                wandb_log(metrics)

        if train_config["test_end_train"]:
            logging.info("Envoi automatique du fichier")
            with open(logdir / "config.yaml", "r") as file:
                print(file)
                test(yaml.safe_load(file))



def send_kaggle(filepath):
    subprocess.run(f"uv run kaggle competitions submit -c 3-md-4040-2026-challenge -f {filepath} -m \"Automatic submission\"",stdout=True,shell=True)
    print("fichier envoyé !")
    


def apply_tta(model, img_batch, tta_operations):
    """
    Applies a list of TTA operations, runs the model, and averages the probabilities.
    Memory-efficient version using a running sum accumulator.
    """
    prob_sum = None
    num_ops = len(tta_operations)
    
    for op in tta_operations:
        # Geometric (Orthogonal)
        if op == 'identity':
            x = img_batch
        elif op == 'hflip':
            x = torch.flip(img_batch, dims=[3])
        elif op == 'vflip':
            x = torch.flip(img_batch, dims=[2])
        elif op == 'rot90':
            x = torch.rot90(img_batch, k=1, dims=[2, 3])
            
        # Geometric (Non-Orthogonal)
        elif op == 'rot15':
            x = TF.rotate(img_batch, angle=15, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        elif op == 'rot30':
            x = TF.rotate(img_batch, angle=30, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        elif op == 'rot45':
            x = TF.rotate(img_batch, angle=45, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        elif op == 'rot60':
            x = TF.rotate(img_batch, angle=60, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            
        # Photometric
        elif op == 'contrast_high':
            # contrast_factor > 1.0 increases contrast
            x = TF.adjust_contrast(img_batch, contrast_factor=1.25)
        elif op == 'contrast_low':
            # contrast_factor < 1.0 decreases contrast
            x = TF.adjust_contrast(img_batch, contrast_factor=0.75)
        else:
            raise ValueError(f"Unknown TTA operation: {op}")
            
        # Forward pass and immediate softmax
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        
        # Accumulate
        if prob_sum is None:
            prob_sum = probs
        else:
            prob_sum += probs
            
    # Average the accumulated probabilities
    avg_probs = prob_sum / num_ops
    return avg_probs

def apply_tta_entropy(model, img_batch, tta_operations, temperature=1.0):
    """
    Applies TTA operations and aggregates probabilities using an Entropy-Weighted Average.
    """
    all_probs = []
    all_entropies = []
    
    for op in tta_operations:
        # Geometric (Orthogonal)
        if op == 'identity':
            x = img_batch
        elif op == 'hflip':
            x = torch.flip(img_batch, dims=[3])
        elif op == 'vflip':
            x = torch.flip(img_batch, dims=[2])
        elif op == 'rot90':
            x = torch.rot90(img_batch, k=1, dims=[2, 3])
            
        # Geometric (Non-Orthogonal)
        elif op == 'rot15':
            x = TF.rotate(img_batch, angle=15, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        elif op == 'rot30':
            x = TF.rotate(img_batch, angle=30, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        elif op == 'rot45':
            x = TF.rotate(img_batch, angle=45, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        elif op == 'rot60':
            x = TF.rotate(img_batch, angle=60, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            
        # Photometric
        elif op == 'contrast_high':
            # contrast_factor > 1.0 increases contrast
            x = TF.adjust_contrast(img_batch, contrast_factor=1.25)
        elif op == 'contrast_low':
            # contrast_factor < 1.0 decreases contrast
            x = TF.adjust_contrast(img_batch, contrast_factor=0.75)
        else:
            raise ValueError(f"Unknown TTA operation: {op}")
            
        # Forward pass
        logits = model(x)
        probs = F_nn.softmax(logits, dim=1)
        
        # Calculate Shannon Entropy: H = -sum(p * log(p))
        # Add 1e-8 for numerical stability to prevent log(0) yielding NaN
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        all_probs.append(probs)
        all_entropies.append(entropy)
        
    # Stack lists into tensors
    # stacked_probs shape: [num_ops, batch_size, num_classes]
    stacked_probs = torch.stack(all_probs, dim=0)
    # stacked_entropies shape: [num_ops, batch_size]
    stacked_entropies = torch.stack(all_entropies, dim=0)
    
    # Compute normalized weights over the num_ops dimension (dim=0)
    # The negative sign ensures lower entropy (higher confidence) gets a higher weight
    weights = F_nn.softmax(-stacked_entropies / temperature, dim=0)
    
    # Broadcast weights: [num_ops, batch_size] -> [num_ops, batch_size, 1]
    weights = weights.unsqueeze(-1)
    
    # Compute the weighted sum
    avg_probs = torch.sum(weights * stacked_probs, dim=0)
    
    return avg_probs

@torch.no_grad()
def extract_model_probabilities(model_path, config_path, use_cuda, tmp_testpath=None,tta_operations=None,tta_entropy=False):
    """
    Loads a single model from its specific config, runs TTA inference, 
    and returns a dictionary of filename -> probability tensor.
    """
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # 1. Load the specific model's configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    model_name = config["model"]["class"]
    model_config = config["model"]
    
    if tta_operations is None:
        test_config = config.get("test", {})
        tta_operations = test_config.get("tta_transforms", ["identity"])
    
    print(f"  - Using TTA operations: {tta_operations}")
    
    num_classes = 86
    
    # 3. Instantiate and load the model
    if "pretrained_path" in model_config and model_config["pretrained_path"]:
        actual_model_class = getattr(models.pretrained_models, model_name)
        model = actual_model_class(
            pretrained_path=model_config["pretrained_path"],
            pretrained=False, 
            num_classes=num_classes, 
        )

        data_cfg = resolve_data_config(model.pretrained_cfg, model=model.get_backbone())
        valid_transform = create_transform(**data_cfg, is_training=False)
        if model_config["pretrained_in_color"]:
            to_rgb = transforms.Lambda(lambda x: x.convert("RGB"))
            valid_transform.transforms.insert(0, to_rgb)

            is_Louis = model_config.get("is_Louis",False)

            if is_Louis:
                valid_transform = [utils.ResizeAndPadToSquare(224)]
                valid_transform.extend([
                    transforms.Grayscale(num_output_channels=3), # Duplication du canal pour le CNN [cite: 205, 208]
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                valid_transform = transforms.Compose(valid_transform)


        #test loader with valid transforms 
        test_loader, input_size, num_classes = data.get_test_dataloaders(
        config, use_cuda, tmp_testpath=tmp_testpath, input_transform = valid_transform
        )

    else:

        #test loader without valid transforms
        valid_transform = v2.Compose([
            v2.Grayscale(), 
            v2.Resize((128, 128), antialias=True),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
        ])
        test_loader, input_size, num_classes = data.get_test_dataloaders(
        config, use_cuda, tmp_testpath=tmp_testpath,input_transform=valid_transform
        )

        actual_model_class = getattr(models.cnn_models, model_name)
        model = actual_model_class(model_config, input_size, num_classes)

        

    model.load_state_dict(torch.load(model_path,map_location=device, weights_only=True))
    model.to(device)
    model.eval()


    # 4. Extract Probabilities
    img_probs = {}
    for img, filenames in test_loader:
        img = img.to(device)

        if tta_entropy:
            batch_probs = apply_tta_entropy(model, img, tta_operations)
        else:
            batch_probs = apply_tta(model, img, tta_operations)
        
        for prob, filename in zip(batch_probs, filenames):
            # MUST move to CPU to prevent VRAM exhaustion
            img_probs[filename] = prob.cpu() 
            
    # Clean up GPU memory before the next model is loaded
    del model
    torch.cuda.empty_cache()
    
    return img_probs


def test_ensemble(ensemble_config, send_kaggle_bool=True):
    """
    Orchestrates the ensemble testing by parsing parallel lists of weights and configs,
    handling OS-level variable expansions for dynamic paths.
    """
    use_cuda = torch.cuda.is_available()
        
    # Extract and expand top-level OS variables
    raw_save_dir = ensemble_config.get("save_dir", "${JOB_WORKSPACE}/logs/ensemble")
    save_dir = os.path.expandvars(raw_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Safely expand tmp_testpath if it exists in the yaml
    raw_tmp_testpath = ensemble_config.get("tmp_testpath", None)
    tmp_testpath = os.path.expandvars(raw_tmp_testpath) if raw_tmp_testpath else None
    
    test_config = ensemble_config.get("test", {})

    global_tta = test_config.get("tta_transforms", ["identity"])
    tta_entropy = test_config.get("tta_entropy",False)

    # Extract the nested YAML structure
    test_config = ensemble_config.get("test", {})
    model_paths = test_config.get("model_path", [])
    config_paths = test_config.get("model_config_path", [])
    
    # Critical Safety Check
    if len(model_paths) != len(config_paths):
        raise ValueError(f"Mismatch in ensemble config: found {len(model_paths)} model paths but {len(config_paths)} config paths.")
    
    # Pair them up
    all_models = list(zip(model_paths, config_paths))
    
    if not all_models:
        print("No models found in the ensemble configuration.")
        return

    print(f"Starting ensemble of {len(all_models)} models...")
    print(f"Using test dataset at: {tmp_testpath}")
    
    ensemble_probs = defaultdict(list)
    
    # 1. Accumulate predictions
    for model_path, config_path in all_models:
        print(f"\nEvaluating: {model_path}")
        # Pass the expanded tmp_testpath down to the extractor
        probs_dict = extract_model_probabilities(model_path, config_path, use_cuda, tmp_testpath,tta_operations=global_tta,tta_entropy=tta_entropy)
        
        for filename, prob in probs_dict.items():
            ensemble_probs[filename].append(prob)
            
    # 2. Average and output
    print("\nAveraging predictions and writing CSV...")
    
    unique_save_path = utils.generate_unique_csv(save_dir, "ensemble_submission")
    
    with open(unique_save_path, "w") as file:
        file.write("imgname,label\n")
        
        for filename, probs_list in ensemble_probs.items():
            # Stack the individual probability vectors: (num_models, num_classes)
            stacked_probs = torch.stack(probs_list)
            # Average across the models
            mean_probs = torch.mean(stacked_probs, dim=0)
            # Take the final highest probability class
            final_pred = torch.argmax(mean_probs).item()
            
            file.write(f"{filename},{final_pred}\n")
            
    print(f"Ensemble evaluation complete. File saved to {unique_save_path}")
    
    if send_kaggle_bool:
        send_kaggle(unique_save_path)

def create_sweep(config):
    """
    Initializes a W&B sweep or connects to an existing one using a single config dictionary, 
    then binds local datasets and launches the agent.
    """
    # 1. Determine Sweep ID or Create Sweep
    sweep_id = config.get("sweep_id")
    
    project = config.get("project")
    entity = config.get("entity")
    count = config.get("count")
    
    if not sweep_id:
        print("No sweep_id found in config. Provisioning a new sweep...")
        # Passes the unified config directly to W&B
        sweep_id = wandb.sweep(sweep=config, project=project, entity=entity)
        print(f"Created new sweep with ID: {sweep_id}")
    else:
        print(f"Connecting to existing sweep_id: {sweep_id}")

    # 2. Extract Data Paths
    tmp_testpath = config.get("tmp_testpath")
    tmp_trainpath = config.get("tmp_trainpath")

    if tmp_testpath:
        print(f"tmp_testpath existe : {tmp_testpath}")
    if tmp_trainpath:
        print(f"tmp_trainpath existe : {tmp_trainpath}")

    # 3. Bind Paths to Training Function
    bound_train_function = partial(
        train_sweep, 
        tmp_trainpath=tmp_trainpath, 
        tmp_testpath=tmp_testpath
    )
    
    # 4. Launch Agent
    wandb.agent(
        sweep_id=sweep_id, 
        project=project,
        entity=entity,
        function=bound_train_function, 
        count=count
    )
if __name__ == "__main__":
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))
    command = sys.argv[2]
    eval(f"{command}(config)")
    
    """
    sweep_configuration = sys.argv[2]
     # Initialize the sweep by passing in the config dictionary
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project)

    # Start the sweep job
    wandb.agent(sweep_id, function=train, count=4)
    """