# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib
import subprocess  # To be able to send the results directly to kaggle 
import datetime  # To enrich the log files and now when the training was launched
import sys
from functools import partial

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
from torchvision import transforms, datasets
from torchvision.transforms import functional as F
import math
import tqdm
import time
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform  # To import pre-trained models, even models already trained at recognizing plankton  
import PIL  # for image pre-processing. I think it's useless but I do want to break the code
from PIL import Image  # Same
import open_clip

# Local imports
from . import data
from . import models
from . import optim
from . import utils

NUM_CLASSES = 86

def is_cuda_usable():
    """
    Check if CUDA is not only available but also actually usable.
    Returns True if CUDA can be used, False otherwise.
    """
    if not torch.cuda.is_available():
        return False
    try:
        # Try to create a tensor on CUDA to verify it's actually usable
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except (RuntimeError, torch.cuda.DeviceError):
        return False

def train_sweep(tmp_testpath=None, tmp_trainpath=None):
    print("Nouvelle run")
    print("Nouvelle run, nouveau code, gros gain")
    use_cuda = is_cuda_usable()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    print(f"We are using {device} for training")

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
        transform = None 

        if "pretrained_path" in model_config:
            logging.info("using a pretrained model") 
            model = timm.create_model(model_config["pretrained_path"], pretrained=True, num_classes=num_classes)
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
            pretrained_in_color = model_config["pretrained_in_color"]#To know if the pretrained_model takes Black and White pictures as inputs or RGB images
        if pretrained_in_color:
            to_rgb = transforms.Lambda(lambda x: x.convert("RGB"))# Does the necessary modifications so that the image now has 3 channels (corresponding to RGB)
            transform.transforms.insert(0, to_rgb) #Adds the duplication of channels at the beginning of transform
   
        # Build the dataloaders
        logging.info("= Building the dataloaders")
        data_config = config["data"]
        batch_size = data_config["batch_size"]

        train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
            data_config, use_cuda, transform=transform
        )

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
            loss = optim.get_focal_loss(config["data"]["trainpath"], device, loss_config["gamma"],tmp_trainpath=tmp_trainpath)
        elif is_weighted:
            loss = optim.get_weighted_loss(lossname, config["data"]["trainpath"],device,tmp_trainpath=tmp_trainpath)
            logging.info("We are using a weighted loss")
        else:
            loss = optim.get_loss(loss_config, config["data"]["trainpath"],device )
            logging.info("We are using a regular (non weighted) loss")

        # Build the optimizer
        logging.info("= Optimizer")
        optim_config = config["optim"]
        optimizer = optim.get_optimizer(optim_config, model.parameters())
        logging.info(f"We are running the latest code ! Yay !")
        # Build the callbacks
        logging_config = config["logging"]
        # Let us use as base logname the class name of the modek
        logname = model_config["class"]
        logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
            logging.info(f"created a logdir at {logdir}")

        logging.info(f"Will be logging into {os.path.abspath(logdir)}")

        # Copy the config file into the logdir
        logdir = pathlib.Path(logdir)
        save_dir = logging_config["save_dir"]
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
            + f"Train : {train_loader.dataset.dataset}\n"
            + f"Validation : {valid_loader.dataset.dataset}"
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
        
        
        try:
            for e in range(train_config["nepochs"]):
                logging.info("Entering a new epoch")
                # Train 1 epoch
                time_before_training = time.time()
                train_loss = utils.train(model, train_loader, loss, optimizer, device,dynamic_display=is_dynamic,batch_size = batch_size)
                time_of_training = (time.time() - time_before_training )/60
                logging.info(f"This epoch took {time_of_training} minutes to train")

                # Test
                time_before_test= time.time()
                test_loss = utils.test(model, valid_loader, loss, device)
                time_of_test = (time.time() - time_before_test) / 60 
                logging.info(f"This test took {time_of_test} minutes to test")

                # Test f1score
                time_before_test= time.time()
                f1score = utils.test_f1score(model, valid_loader, num_classes, device)
                time_of_test = (time.time() - time_before_test) / 60 
                logging.info(f"This test took {time_of_test} minutes to test")


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

                # Update the dashboard
                metrics = {"train_CE": train_loss, "test_CE": test_loss, "f1score": f1score}
                if wandb_log is not None:
                    logging.info("Logging on wandb")
                    wandb_log(metrics)

        except BaseException as e:
            logging.warning(f"Arrêt Hyperband intercepté (Type: {type(e).__name__}). Destruction des workers PyTorch...")
        
        # 1. On tue violemment les processus de chargement de données
            if 'dataloader' in locals():
                if hasattr(dataloader, '_iterator') and dataloader._iterator is not None:
                    dataloader._iterator._shutdown_workers() # Méthode interne de PyTorch pour forcer le kill
            del dataloader
        
            # 2. On laisse l'exception remonter pour que l'agent W&B clôture le run proprement
            raise e

            if train_config["test_end_train"]:
                logging.info("Envoi automatique du fichier")
                with open(logdir / "config.yaml", "r") as file:
                    print(file)
                    ###### ADD THE NECESSARY STUFF TO THE TEST CONFIG FILE FOR EASIER TESTING !!!!
                    test(yaml.safe_load(file))


def train(config):
    print("Debut du train")
    print("Are we running an old version ?")
    use_cuda = is_cuda_usable()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    print(f"We are using {device} for training")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
        
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    train_config = config["train"]

    num_classes = NUM_CLASSES

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    transform = None

    if "pretrained_path" in model_config:
        logging.info("using a pretrained model") 
        model = timm.create_model(model_config["pretrained_path"], pretrained=True, num_classes=num_classes)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        pretrained_in_color = model_config["pretrained_in_color"]#To know if the pretrained_model takes Black and White pictures as inputs or RGB images
        if pretrained_in_color:
            to_rgb = transforms.Lambda(lambda x: x.convert("RGB"))# Does the necessary modifications so that the image now has 3 channels (corresponding to RGB)
            transform.transforms.insert(0, to_rgb) #Adds the duplication of channels at the beginning of transform
   
   # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]
    batch_size = data_config["batch_size"]

    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda, transform=transform
    )

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
    if is_weighted:
        loss = optim.get_weighted_loss(loss_config["lossname"], config["data"]["trainpath"],device )
        logging.info("We are using a weighted loss")
    else:
        loss = optim.get_loss(loss_config, config["data"]["trainpath"],device )
        logging.info("We are using a regular (non weighted) loss")

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())
    logging.info(f"We are running the latest code ! Yay !")
    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
        logging.info(f"created a logdir at {logdir}")

    logging.info(f"Will be logging into {os.path.abspath(logdir)}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    save_dir = logging_config["save_dir"]
    with open(logdir / "config.yaml", "w") as file:
        ###### ADD THE NECESSARY STUFF TO THE TEST CONFIG FILE FOR EASIER TESTING !!!!
        yaml.dump(config, file)
        file.write(f"test:\n    model_path: {os.path.abspath(logdir)}/best_model.pt\n    save_dir: {save_dir}")

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
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {valid_loader.dataset.dataset}"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
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

        # Test
        time_before_test= time.time()
        test_loss = utils.test(model, valid_loader, loss, device)
        time_of_test = (time.time() - time_before_test) / 60 
        logging.info(f"This test took {time_of_test} minutes to test")


        updated = model_checkpoint.update(test_loss)
        logging.info(
            "[%d/%d] Test loss : %.3f %s"
            % (
                e,
                train_config["nepochs"],
                test_loss,
                "[>> BETTER <<]" if updated else "",
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

        if updated:
            logging.info(f"We are logging an artifact !")
            artifact = wandb.Artifact(name="best-model",type ="model",metadata={"loss" : loss, "epoch" : e})
            artifact.add_file(model_checkpoint_loss.savepath)
            wandb.log_artifact(artifact)

        # Update the dashboard
        metrics = {"train_CE": train_loss, "test_CE": test_loss}
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log(metrics)

    if train_config["test_end_train"]:
        logging.info("Envoi automatique du fichier")
        with open(logdir / "config.yaml", "r") as file:
            print(file)
            ###### ADD THE NECESSARY STUFF TO THE TEST CONFIG FILE FOR EASIER TESTING !!!!
            test(yaml.safe_load(file),tmp_testpath=tmp_testpath)


def send_kaggle(filepath):
    
    subprocess.run(f"uv run kaggle competitions submit -c 3-md-4040-2026-challenge -f {filepath} -m \"Automatic submission\"",stdout=True,shell=True)
    print("fichier envoyé !")
    
@torch.no_grad()
def test(config,send_kaggle_bool=True,tmp_testpath=None):
    """
    This function should take the model we want to test ie probably the best model 
    0.jpg, 1 
    1.jpg, 10 
    ...
    121427.jpg, 37

    The name of the file is still to be discussed but we can imagine taking the jobid of the slurm submission
    """
    use_cuda = is_cuda_usable()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    print(f"Device used {device}")
    print("Yay on utilise la nouvelle fonction de test !")

    # ------------------------------------------------------------------
    # Optional Test-Time Augmentation (TTA)
    # ------------------------------------------------------------------
    test_cfg = config.get("test", {}) if isinstance(config, dict) else {}
    use_tta = bool(test_cfg.get("tta", False))
    if use_tta:
        print("Test-time augmentation (TTA) is ENABLED (using flips).")
    else:
        print("Test-time augmentation (TTA) is DISABLED.")

    model_name = config["model"]["class"]
    model_path_list = config["test"]["model_path"]
    for model_path in model_path_list:
        print(f"We are currently testing the model at {model_path}")
        save_dir = config["test"]["save_dir"]
        unique_save_path = utils.generate_unique_csv(save_dir,model_name)
        print(f"unique save path is {unique_save_path}")
        
        model_config = config["model"]
        
        # Check if model requires RGB input (e.g., PlanktonMobileNet uses pretrained MobileNetV2)
        requires_rgb = model_name == "PlanktonMobileNet"
        test_transform = None
        
        if requires_rgb:
            print("PlanktonMobileNet detected: setting up RGB transform using timm data config")
            # Create a lightweight model (pretrained=False) just to get the config structure
            # This is much faster than loading pretrained weights
            # PlanktonMobileNet uses mobilenetv2_140 from timm
            temp_model = timm.create_model('mobilenetv2_140', pretrained=False, num_classes=86)
            data_cfg = resolve_data_config(temp_model.pretrained_cfg, model=temp_model)
            #test_transform = create_transform(**data_cfg, is_training=False)
            test_transform = transforms.Compose([utils.ResizeAndPadToSquare(224),
            transforms.Grayscale(num_output_channels=3), # Duplication du canal pour le CNN [cite: 205, 208]
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
            del temp_model  # Clean up immediately
            
            # Add RGB conversion at the beginning of the transform pipeline
            # This converts grayscale PIL images to RGB before other transforms
            #to_rgb = transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x)
            #test_transform.transforms.insert(0, to_rgb)
        
        test_loader, input_size, num_classes = data.get_test_dataloaders(
            config, use_cuda, tmp_testpath=tmp_testpath, transform=test_transform
        ) 
        
        model = eval(f"models.cnn_models.{model_name}({model_config} ,{input_size},{num_classes})")
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        with open(unique_save_path,"w") as file:
            model.eval()
            print(f"fichier crée à l'adresse : {unique_save_path}")
            i = 0
            file.write("imgname,label \n")
            for img, filenames in test_loader:
                img = img.to(device)

                # ----------------------------------------------------------
                # Forward pass with optional TTA
                # ----------------------------------------------------------
                if use_tta:
                    logits_list = []
                    
                    # Original
                    logits_list.append(model(img))
                    
                    # Horizontal flip
                    logits_list.append(model(torch.flip(img, dims=[3])))

                    # Vertical flip
                    logits_list.append(model(torch.flip(img, dims=[2])))
                    #Rotations: ±15°, ±30°
                    for angle in [15, -15, 30, -30]:
                        img_rotated = F.rotate(img, angle, interpolation=F.InterpolationMode.BILINEAR, fill=0)
                        logits_list.append(model(img_rotated))
                    
                    # Translations: small shifts in x and y directions
                    # Translation parameters: (tx, ty) in pixels
                    # For 128x128 images, translate by ~5% (6 pixels)
                    translate_x, translate_y = 6, 6
                    # for tx, ty in [(translate_x, 0), (-translate_x, 0), (0, translate_y), (0, -translate_y)]:
                    #     # Convert pixel translation to affine matrix parameters
                    #     # affine matrix: [[1, 0, tx], [0, 1, ty]]
                    #     img_translated = F.affine(
                    #         img, angle=0, translate=(tx, ty), scale=1.0, 
                    #         shear=0, interpolation=F.InterpolationMode.BILINEAR, fill=0
                    #     )
                    #     logits_list.append(model(img_translated))
                    
                    # Shear: small shearing transformations
                    # Shear angles in degrees
                    # for shear_x, shear_y in [(5, 0), (-5, 0), (0, 5), (0, -5)]:
                    #     img_sheared = F.affine(
                    #         img, angle=0, translate=(0, 0), scale=1.0,
                    #         shear=(shear_x, shear_y), interpolation=F.InterpolationMode.BILINEAR, fill=0
                    #     )
                    #     logits_list.append(model(img_sheared))
                    
                    # Average logits over all augmented versions
                    logits = torch.stack(logits_list, dim=0).mean(dim=0)
                else:
                    logits = model(img)

                preds = torch.argmax(logits, dim=1)
                for pred, filename in zip(preds,filenames):
                    file.write(f"{filename}, {pred.item()} \n")
                    print(filename)
                    i += 1
        print("Fin du test.")
        if send_kaggle_bool:
            send_kaggle(unique_save_path)
    return None


@torch.no_grad()
def test_ensemble(config_paths, send_kaggle_bool=True, tmp_testpath=None, ensemble_method="average_logits"):
    """
    Ensemble inference using multiple models, each with its own config.yaml file.
    
    Args:
        config_paths: List of paths to config.yaml files, one per model
        send_kaggle_bool: Whether to automatically submit to Kaggle
        tmp_testpath: Optional temporary test path override
        ensemble_method: How to combine predictions. Options:
            - "average_logits": Average logits from all models (soft voting)
            - "majority_vote": Majority voting on predictions (hard voting)
    
    The function:
    1. Loads each model from its config file
    2. Runs inference with each model (with optional TTA per model)
    3. Combines predictions using the specified method
    4. Writes final predictions to CSV
    """
    use_cuda = is_cuda_usable()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    print(f"Device used {device}")
    print(f"Running ensemble inference with {len(config_paths)} models")
    print(f"Ensemble method: {ensemble_method}")
    
    # Load all configs and models
    configs = []
    loaded_models = []
    test_loaders = []
    model_names = []
    
    for i, config_path in enumerate(config_paths):
        print(f"\n--- Loading model {i+1}/{len(config_paths)} from {config_path} ---")
        config = yaml.safe_load(open(config_path, "r"))
        configs.append(config)
        
        model_name = config["model"]["class"]
        model_names.append(model_name)
        print(f"Model name: {model_name}")
        
        # Get model path from config
        model_path_list = config["test"]["model_path"]
        if isinstance(model_path_list, list):
            model_path = model_path_list[0]  # Use first model path if list
        else:
            model_path = model_path_list
        
        # Check if model requires RGB input
        requires_rgb = model_name == "PlanktonMobileNet"
        test_transform = None
        
        if requires_rgb:
            print("PlanktonMobileNet detected: setting up RGB transform")
            temp_model = timm.create_model('mobilenetv2_140', pretrained=False, num_classes=86)
            data_cfg = resolve_data_config(temp_model.pretrained_cfg, model=temp_model)
            test_transform = transforms.Compose([
                utils.ResizeAndPadToSquare(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            del temp_model
        
        # Build test dataloader for this model
        test_loader, input_size, num_classes = data.get_test_dataloaders(
            config, use_cuda, tmp_testpath=tmp_testpath, transform=test_transform
        )
        test_loaders.append(test_loader)
        
        # Build and load model
        model_config = config["model"]
        model = eval(f"models.cnn_models.{model_name}({model_config}, {input_size}, {num_classes})")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(device)
        model.eval()
        loaded_models.append(model)
        print(f"Model {i+1} loaded successfully")
    
    # Verify all models have the same number of classes
    num_classes_list = [len(loader.dataset.classes) for loader in test_loaders]
    if len(set(num_classes_list)) > 1:
        raise ValueError(f"Models have different numbers of classes: {num_classes_list}")
    num_classes = num_classes_list[0]
    
    # Get save directory from first config
    save_dir = configs[0]["test"]["save_dir"]
    ensemble_name = "_".join(model_names) + "_ensemble"
    unique_save_path = utils.generate_unique_csv(save_dir, ensemble_name)
    print(f"\nSaving ensemble predictions to: {unique_save_path}")
    
    # Get TTA settings for each model
    use_tta_list = []
    for config in configs:
        test_cfg = config.get("test", {}) if isinstance(config, dict) else {}
        use_tta = bool(test_cfg.get("tta", False))
        use_tta_list.append(use_tta)
        if use_tta:
            print(f"Model {config['model']['class']}: TTA ENABLED")
        else:
            print(f"Model {config['model']['class']}: TTA DISABLED")
    
    # Run ensemble inference
    # Note: We assume all dataloaders:
    # 1. Iterate in the same order (same test path)
    # 2. Have the same batch size (recommended: use same batch_size in all configs)
    # 3. Process the same test images (same testpath in configs)
    # We iterate through them in parallel.
    with open(unique_save_path, "w") as file:
        file.write("imgname,label \n")
        
        # Create iterators for all dataloaders
        dataloader_iters = [iter(loader) for loader in test_loaders]
        
        batch_idx = 0
        while True:
            try:
                # Get batches from all dataloaders in parallel
                batches = []
                filenames = None
                for loader_iter in dataloader_iters:
                    img_batch, fnames = next(loader_iter)
                    img_batch = img_batch.to(device)
                    batches.append(img_batch)
                    if filenames is None:
                        filenames = fnames  # Use filenames from first loader
                
                # Collect logits from all models
                all_logits = []
                
                for model, img_batch, use_tta in zip(loaded_models, batches, use_tta_list):
                    if use_tta:
                        logits_list = []
                        # Original
                        logits_list.append(model(img_batch))
                        # Horizontal flip
                        logits_list.append(model(torch.flip(img_batch, dims=[3])))
                        # Vertical flip
                        logits_list.append(model(torch.flip(img_batch, dims=[2])))
                        # Average logits over all augmented versions
                        logits = torch.stack(logits_list, dim=0).mean(dim=0)
                    else:
                        logits = model(img_batch)
                    
                    all_logits.append(logits)
                
                # Combine predictions
                if ensemble_method == "average_logits":
                    # Average logits (soft voting)
                    ensemble_logits = torch.stack(all_logits, dim=0).mean(dim=0)
                    preds = torch.argmax(ensemble_logits, dim=1)
                elif ensemble_method == "majority_vote":
                    # Majority voting (hard voting)
                    all_preds = [torch.argmax(logits, dim=1) for logits in all_logits]
                    # Stack predictions and take mode
                    preds_stack = torch.stack(all_preds, dim=0)  # Shape: (num_models, batch_size)
                    # Use mode (most common prediction) for each sample
                    preds, _ = torch.mode(preds_stack, dim=0)
                else:
                    raise ValueError(f"Unknown ensemble_method: {ensemble_method}. Use 'average_logits' or 'majority_vote'")
                
                # Write predictions
                for pred, filename in zip(preds, filenames):
                    file.write(f"{filename}, {pred.item()} \n")
                
                batch_idx += 1
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx} batches")
                    
            except StopIteration:
                # All dataloaders exhausted
                break
    
    print("Ensemble inference finished.")
    if send_kaggle_bool:
        send_kaggle(unique_save_path)
    
    return None


@torch.no_grad()
def test_zero_shot_clip(config, send_kaggle_bool=True, tmp_testpath=None, tmp_trainpath=None):
    """
    Zero-shot classification using OpenCLIP LAION models:
    - Convert each class name -> text embedding
    - Convert each test image -> image embedding
    - Compare cosine similarity and pick the highest score
    """
    use_cuda = is_cuda_usable()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    print(f"Device used {device}")
    print("Running zero-shot classification with OpenCLIP")

    # ------------------------------------------------------------------
    # 1. Load CLIP model & preprocess from config
    # ------------------------------------------------------------------
    clip_cfg = config.get("clip", {})
    model_name = clip_cfg.get("model_name", "ViT-B-32")
    pretrained = clip_cfg.get("pretrained", "laion2b_s34b_b79k")
    template = clip_cfg.get("template", "a photo of a {}")

    print(f"Loading OpenCLIP model '{model_name}' with weights '{pretrained}'")
    clip_model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    clip_model.to(device)
    clip_model.eval()

    # ------------------------------------------------------------------
    # 2. Build the list of class names from the training folder
    # ------------------------------------------------------------------
    data_config = config["data"]
    if tmp_trainpath:
        trainpath = tmp_trainpath
    else:
        trainpath = data_config["trainpath"]

    print(f"Loading class names from '{trainpath}'")
    base_dataset = datasets.ImageFolder(root=trainpath)
    raw_classes = base_dataset.classes  # e.g. ['class_a', 'class_b', ...]

    # Turn folder names into more natural language prompts
    classnames = [template.format(c.replace("_", " ")) for c in raw_classes]
    print(f"Number of classes for zero-shot: {len(classnames)}")

    text_tokens = tokenizer(classnames).to(device)
    text_features = clip_model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # 3. Build the test dataloader using CLIP's preprocess
    # ------------------------------------------------------------------
    test_loader, input_size, num_classes = data.get_test_dataloaders(
        config, use_cuda, tmp_testpath=tmp_testpath, transform=preprocess
    )

    # ------------------------------------------------------------------
    # 4. Run zero-shot prediction and write Kaggle CSV
    # ------------------------------------------------------------------
    model_name_for_log = clip_cfg.get("save_name", f"{model_name}_{pretrained}")
    save_dir = config["logging"]["save_dir"]
    unique_save_path = utils.generate_unique_csv(save_dir, model_name_for_log)
    print(f"Saving zero-shot predictions to: {unique_save_path}")

    with open(unique_save_path, "w") as file:
        clip_model.eval()
        file.write("imgname,label \n")

        for imgs, filenames in test_loader:
            imgs = imgs.to(device)

            image_features = clip_model.encode_image(imgs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Similarity scores (scaled like original CLIP implementation)
            logits = 100.0 * image_features @ text_features.T
            preds = torch.argmax(logits, dim=1)

            for pred, filename in zip(preds, filenames):
                file.write(f"{filename}, {pred.item()} \n")

    print("Zero-shot CLIP inference finished.")
    if send_kaggle_bool:
        send_kaggle(unique_save_path)

    return None


@torch.no_grad()
def test_supervised_clip(config, send_kaggle_bool=False, tmp_testpath=None, tmp_trainpath=None):
    """
    Supervised CLIP classification using training-image prototypes:
    - Encode each training image with CLIP
    - Build one prototype per class (mean embedding)
    - For each test image, encode with CLIP and pick the class with highest cosine similarity
    """
    use_cuda = is_cuda_usable()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    print(f"Device used {device}")
    print("Running supervised prototype-based classification with OpenCLIP")

    # ------------------------------------------------------------------
    # 1. Load CLIP model & preprocess from config
    # ------------------------------------------------------------------
    clip_cfg = config.get("clip", {})
    model_name = clip_cfg.get("model_name", "ViT-B-32")
    pretrained = clip_cfg.get("pretrained", "laion2b_s34b_b79k")

    print(f"Loading OpenCLIP model '{model_name}' with weights '{pretrained}'")
    clip_model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    clip_model.to(device)
    clip_model.eval()

    data_config = config["data"]
    if tmp_trainpath:
        trainpath = tmp_trainpath
    else:
        trainpath = data_config["trainpath"]

    print(f"Building training dataset from '{trainpath}' for CLIP prototypes")
    train_dataset = datasets.ImageFolder(root=trainpath, transform=preprocess)
    num_classes = len(train_dataset.classes)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=use_cuda,
    )

    # ------------------------------------------------------------------
    # 2. Compute class prototypes (mean normalized embedding per class)
    # ------------------------------------------------------------------
    prototypes = None
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        feats = clip_model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        if prototypes is None:
            prototypes = torch.zeros(num_classes, feats.size(-1), device=device)

        for c in labels.unique():
            mask = labels == c
            if mask.any():
                prototypes[c] += feats[mask].sum(dim=0)
                counts[c] += mask.sum()

    # Avoid division by zero; classes without samples keep zero prototype
    counts_safe = counts.clamp(min=1).unsqueeze(-1)
    prototypes = prototypes / counts_safe
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # 3. Build the test dataloader using CLIP's preprocess
    # ------------------------------------------------------------------
    test_loader, input_size, _ = data.get_test_dataloaders(
        config, use_cuda, tmp_testpath=tmp_testpath, transform=preprocess
    )

    # ------------------------------------------------------------------
    # 4. Run prototype-based prediction and write Kaggle CSV
    # ------------------------------------------------------------------
    model_name_for_log = clip_cfg.get("save_name", f"{model_name}_{pretrained}_supervised")
    save_dir = config["logging"]["save_dir"]
    unique_save_path = utils.generate_unique_csv(save_dir, model_name_for_log)
    print(f"Saving supervised CLIP predictions to: {unique_save_path}")

    with open(unique_save_path, "w") as file:
        clip_model.eval()
        file.write("imgname,label \n")

        for imgs, filenames in test_loader:
            imgs = imgs.to(device)

            image_features = clip_model.encode_image(imgs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features @ prototypes.T
            preds = torch.argmax(logits, dim=1)

            for pred, filename in zip(preds, filenames):
                file.write(f"{filename}, {pred.item()} \n")

    print("Supervised CLIP prototype inference finished.")
    if send_kaggle_bool:
        send_kaggle(unique_save_path)

    return None

def create_sweep(sweep_config):
    project = config["project"]
    entity = config["entity"]
    count = config["count"]
    sweep_id = wandb.sweep(sweep = sweep_config,project = project, entity = entity)
    wandb.agent(sweep_id = sweep_id, function = train_sweep, count= count)

def launch_agent(config):
    sweep_id = config["first_sweep_id"]
    print(sweep_id)
    if "tmp_testpath" in config:
        print(f"tmp_testpath existe : {config['tmp_testpath']}")
        tmp_testpath = config["tmp_testpath"]
    if "tmp_trainpath" in config:
        print(f"tmp_trainpath existe : {config['tmp_trainpath']}")
        tmp_trainpath = config["tmp_trainpath"]
    bound_train_function = partial(train_sweep, tmp_trainpath=tmp_trainpath, tmp_testpath=tmp_testpath)
    wandb.agent(sweep_id=sweep_id, function=bound_train_function)

if __name__ == "__main__":
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test|test_ensemble>")
        logging.error(f"For test_ensemble: {sys.argv[0]} config1.yaml,config2.yaml,... test_ensemble [ensemble_method]")
        sys.exit(-1)

    config_path = sys.argv[1]
    command = sys.argv[2]
    
    # Handle ensemble mode specially
    if command == "test_ensemble":
        # For ensemble, config_path should be comma-separated list of config files
        config_paths = [path.strip() for path in config_path.split(",")]
        logging.info(f"Running ensemble with {len(config_paths)} models")
        for i, path in enumerate(config_paths):
            logging.info(f"  Model {i+1}: {path}")
        
        # Optional ensemble method (default: average_logits)
        ensemble_method = sys.argv[3] if len(sys.argv) > 3 else "average_logits"
        test_ensemble(config_paths, ensemble_method=ensemble_method)
    else:
        logging.info("Loading {}".format(config_path))
        config = yaml.safe_load(open(config_path, "r"))
        eval(f"{command}(config)")
    
    """
    sweep_configuration = sys.argv[2]
     # Initialize the sweep by passing in the config dictionary
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project)

    # Start the sweep job
    wandb.agent(sweep_id, function=train, count=4)
    """