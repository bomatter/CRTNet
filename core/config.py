import os
import yaml
import subprocess

from ml_collections import ConfigDict


def create_config(args):
    """
    Returns a ConfigDict. If args.config_file is not specified, default values are used. Further fileds in args are used instead of defaults.

    Arguments:
        args: Commandline arguments created with argparse.
    """

    # Load or create new config
    if args.config is not None:
        with open(args.config) as f:
            cfg = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    else:
        cfg = ConfigDict()

    # Set default values if not specified in the loaded config file and assert that required arguments were specified.
    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint
    elif not hasattr(cfg, "checkpoint"):
        cfg.checkpoint = None

    if args.annotations is not None:
        cfg.annotations = args.annotations
    assert(hasattr(cfg, "annotations")), "No annotations file was specified. Specify it in the config file or via commandline argument"
    
    if args.imagedir is not None:
        cfg.imagedir = args.imagedir
    assert(hasattr(cfg, "imagedir")), "No imagedir was specified. Specify it in the config file or via commandline argument"
    
    if args.test_annotations is not None:
        cfg.test_annotations = args.test_annotations
    elif not hasattr(cfg, "test_annotations"):
        cfg.test_annotations = None
    
    if args.test_imagedir is not None:
        cfg.test_imagedir = args.test_imagedir
    elif not hasattr(cfg, "test_imagedir"):
        cfg.test_imagedir = None

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    elif not hasattr(cfg, "batch_size"):
        cfg.batch_size = 16

    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    elif not hasattr(cfg, "learning_rate"):
        cfg.learning_rate = 1e-5

    # add hash of last git commit to config if available
    try:
        cfg.git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except:
        print("Could not save git state to config.")

    return cfg
    

def save_config(cfg, savedir):
    """
    Saves the ConfigDict as a yaml config file to savedir.

    Arguments:
        cfg: ConfigDict to be saved
        savedir: Path to folder where the config file should be saved.
    """

    with open(os.path.join(savedir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)