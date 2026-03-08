"""
This file is temporay and is used to keep the functions I think I can throw away but I fear that I need to use again someday
"""

def create_sweep(sweep_config):
    print("Creating a sweep")
    project = config["project"]
    entity = config["entity"]
    count = config["count"]
    sweep_id = wandb.sweep(sweep = sweep_config,project = project, entity = entity)
    wandb.agent(sweep_id=sweep_id, function=train_sweep)

def launch_agent(config, sweep_id=None):
    """
    This function is only used to connect to an already running sweep 
    """
    if not sweep_id:
        sweep_id = config["first_sweep_id"]
    print(sweep_id)
    
    # .get() returns None if the key is missing, preventing UnboundLocalError
    tmp_testpath = config.get("tmp_testpath")
    tmp_trainpath = config.get("tmp_trainpath")

    if tmp_testpath:
        print(f"tmp_testpath existe : {tmp_testpath}")
    if tmp_trainpath:
        print(f"tmp_trainpath existe : {tmp_trainpath}")
        
    bound_train_function = partial(train_sweep, tmp_trainpath=tmp_trainpath, tmp_testpath=tmp_testpath)
    wandb.agent(sweep_id=sweep_id, function=bound_train_function)

def train(config):
    """
    This launches a function without a wandb sweep. It is preferred to use a wandb sweep 
    """
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


"""
old optim.py
"""

"""
old testing logic without ensemble
"""
    
@torch.no_grad()
def test(config, send_kaggle_bool=True, tmp_testpath=None):
    """
    Evaluates the model and generates a CSV for Kaggle submission.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    print(f"Device used: {device}")
    print("Yay on utilise la nouvelle fonction de test !")

    model_name = config["model"]["class"]
    model_path_list = config["test"]["model_path"]
    model_config = config["model"]
    
    if "pretrained_path" in model_config and model_config["pretrained_path"]:
        csv_base_name = model_config["pretrained_path"].replace("/", "_").replace(":", "_")
    else:
        csv_base_name = model_name

    # Properly expand the shell variables
    save_dir = os.path.expandvars(config["test"]["save_dir"])
    
    # Optional performance fix: Load the test data ONCE outside the loop 
    # since it doesn't change between model evaluations.
    test_loader, input_size, num_classes = data.get_test_dataloaders(config, use_cuda, tmp_testpath=tmp_testpath)
    
    for model_path in model_path_list:
        print(f"We are currently testing the model at {model_path}")
        unique_save_path = utils.generate_unique_csv(save_dir, csv_base_name)
        print(f"Unique save path is {unique_save_path}")
        
        if "pretrained_path" in model_config and model_config["pretrained_path"]:
            # Dynamically fetch the class from the correct module
            actual_model_class = getattr(models.pretrained_models, model_name)
            model = actual_model_class(
                pretrained_path=model_config["pretrained_path"],
                pretrained=False, 
                num_classes=num_classes, 
            )
        else:
            # Dynamically fetch the class for custom CNNs
            actual_model_class = getattr(models.cnn_models, model_name)
            model = actual_model_class(model_config, input_size, num_classes)

        # Removed the eval() line entirely.

        # Load weights and push to device
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(device)
        model.eval()

        with open(unique_save_path, "w") as file:
            print(f"Fichier créé à l'adresse : {unique_save_path}")
            file.write("imgname,label\n")
            
            for img, filenames in test_loader:
                img = img.to(device)
                logits = model(img)
                preds = torch.argmax(logits, dim=1) 
                
                for pred, filename in zip(preds, filenames):
                    file.write(f"{filename},{pred.item()}\n")
                    
        print(f"Fin du test pour {model_path}.")
        
        if send_kaggle_bool:
            send_kaggle(unique_save_path)
            
    return None