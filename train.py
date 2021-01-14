import argparse
import pathlib
import yaml

import torch, torchvision
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from test import test
from core.config import create_config, save_config
from core.dataset import COCODataset
from core.metrics import AccuracyLogger


## Initialization
#

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="(Optional) Path to config file. If additional commandline options are provided, they are used to modify the specifications in the conifg file.")
parser.add_argument("--outdir", type=str, default="output", help="Path to output folder (will be created if it does not exist).")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint from which to continue training.")
parser.add_argument("--annotations", type=str, help="Path to COCO-style annotations file.")
parser.add_argument("--imagedir", type=str, help="Path to images folder w.r.t. which filenames are specified in the annotations.")

parser.add_argument("--test_annotations", type=str, help="Path to COCO-style annotations file for model evaluation.")
parser.add_argument("--test_imagedir", type=str, help="Path to images folder w.r.t. which filenames are specified in the annotations for model evaluation.")
parser.add_argument("--test_frequency", type=int, default=1, help="Evaluate model on test data every __ epochs.")

parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train.")
parser.add_argument("--batch_size", type=int, help="Batchsize to use for training.")
parser.add_argument("--print_batch_metrics", type=bool, default=False, help="Set True to print metrics for every batch.")
args = parser.parse_args()


# load config or create a new one
cfg = create_config(args)
save_config(cfg, args.outdir)
