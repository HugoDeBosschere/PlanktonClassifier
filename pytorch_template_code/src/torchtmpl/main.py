# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib
import subprocess 
import datetime 
from functools import partial

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
from torchvision import transforms
import tqdm
import time
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform 
import PIL 
from PIL import Image 

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
        train_transform = None
        valid_transform = None 

        if "pretrained_path" in model_config:
            logging.info("using a pretrained model") 
            
            
            # Conditionally freeze the backbone
            freeze_pretrained = model_config.get("freeze_pretrained", False)
            if freeze_pretrained:
                logging.info("Freezing the pre-trained backbone.")
                for param in model.parameters():
                    param.requires_grad = False
            else:
                logging.info("Fine-tuning the entire model (backbone unfrozen).")

            if "old_model_path" in model_config:
                old_model_path = model_config["old_model_path"]
                logging.info(f"Loading model at {old_model_path}")
                model.load_state_dict(torch.load(model_config["old_model_path"],weights_only=True))
                classifier_module = model.get_classifier() 
                for param in classifier_module.parameters(): #unfreezing the parameters of the classifier
                    param.requires_grad = True
            else:
                model.reset_classifier(num_classes = NUM_CLASSES)
            
            train_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model),is_training=True,scale=(0.8, 1.0),color_jitter=0)
            valid_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model),is_training = False)
        
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
            loss = optim.get_weighted_loss(lossname, class_counts, device)
            logging.info("We are using a weighted loss")
        else:
            loss = optim.get_loss(loss_config, config["data"]["trainpath"], device)
            logging.info("We are using a regular (non weighted) loss")

        # Build the optimizer
        logging.info("= Optimizer")
        optim_config = config["optim"]
        optimizer = optim.get_optimizer(optim_config, filter(lambda p: p.requires_grad, model.parameters()))
        logging.info(f"We are running the latest code ! Yay !")
        # Build the callbacks
        logging_config = config["logging"]
        # The logname is the pretrained path if it exists, the name of the base model if it doesn't
        if "pretrained_path" in model_config and model_config["pretrained_path"]:
            logname = model_config["pretrained_path"].replace("/", "_").replace(":", "_")
        else:
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
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    print(f"Device used {device}")
    print("Yay on utilise la nouvelle fonction de test !")

    model_name = config["model"]["class"]
    model_path_list = config["test"]["model_path"]
    model_config = config["model"]
    
    if "pretrained_path" in model_config and model_config["pretrained_path"]:
        csv_base_name = model_config["pretrained_path"].replace("/", "_").replace(":", "_")
    else:
        csv_base_name = model_name

    # Hoisted outside the loop
    save_dir = config["test"]["save_dir"] 
    
    for model_path in model_path_list:
        print(f"We are currently testing the model at {model_path}")
        unique_save_path = utils.generate_unique_csv(save_dir,csv_base_name)
        print(f"unique save path is {unique_save_path}")
        test_loader, input_size, num_classes = data.get_test_dataloaders(config, use_cuda,tmp_testpath=tmp_testpath)
        
        model_config = config["model"]

        if "pretrained_path" in model_config and model_config["pretrained_path"]:
            model = timm.create_model(model_config["pretrained_path"], pretrained=False, num_classes=num_classes)
        else:
            model_class = getattr(models.cnn_models, model_name)
            model = model_class(model_config, input_size, num_classes)

        model = eval(f"models.cnn_models.{model_name}({model_config} ,{input_size},{num_classes})")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(device)

        with open(unique_save_path,"w") as file:
            model.eval()
            print(f"fichier crée à l'adresse : {unique_save_path}")
            i = 0
            file.write("imgname,label \n")
            for img, filenames in test_loader:
                img = img.to(device)
                logits = model(img)
                preds = torch.argmax(logits,dim=1) 
                for pred, filename in zip(preds,filenames):
                    file.write(f"{filename}, {pred.item()} \n")
                    print(filename)
                    i += 1
        print("Fin du test.")
        if send_kaggle_bool:
            send_kaggle(unique_save_path)
    return None

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