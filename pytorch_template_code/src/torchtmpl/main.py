# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib
import subprocess # To be able to send the results directly to kaggle 
import datetime # To enrich the log files and now when the training was launched
import sys

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
import tqdm
import time 

# Local imports
from . import data
from . import models
from . import optim
from . import utils


def train(config):
    print("Debut du train")
    print("Are we running an old version ?")
    use_cuda = torch.cuda.is_available()
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

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]
    batch_size = data_config["batch_size"]

    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    if "old_model_path" in model_config:
        old_model_path = model_config["old_model_path"]
        logging.info(f"Loading model at {old_model_path}")
        model.load_state_dict(torch.load(model_config["old_model_path"],weights_only=True))
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    is_weighted = train_config["is_weighted"]
    if is_weighted:
        loss = optim.get_weighted_loss(train_config["loss"], config["data"]["trainpath"],device )
        logging.info("We are using a weighted loss")
    else:
        loss = optim.get_loss(train_config["loss"])
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
    with open(logdir / "config.yaml", "w") as file:
        ###### ADD THE NECESSARY STUFF TO THE TEST CONFIG FILE FOR EASIER TESTING !!!!
        yaml.dump(config, file)
        file.write(f"test:\n    model_path: {os.path.abspath(logdir)}/best_model.pt\n    save_dir: /usr/users/sdim/sdim_9/DeepLearning/pytorch_template_code/test")

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
            artifact.add_file(model_checkpoint.savepath)
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
            test(yaml.safe_load(file))


def send_kaggle(filepath):
    print("Envoi du fichier")
    subprocess.run(f"kaggle competitions submit -c 3-md-4040-2026-challenge -f {filepath} -m \"Automatic submission\"",stdout=True,shell=True)

@torch.no_grad()
def test(config,send_kaggle_bool=True):
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
    model_path = config["test"]["model_path"]
    print(f"We are currently testing the model at {model_path}")
    save_dir = config["test"]["save_dir"]
    unique_save_path = utils.generate_unique_csv(save_dir,model_name)
    print(f"unique save path is {unique_save_path}")
    test_loader, input_size, num_classes = data.get_test_dataloaders(config, use_cuda)
    
    model_config = config["model"]

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