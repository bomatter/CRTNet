import os
import argparse
import datetime
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
from core.model import Model
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
parser.add_argument("--learning_rate", type=float, help="Learning rate to use for training.")
parser.add_argument("--save_frequency", type=int, default=1, help="Save model checkpoint every __ epochs.")
parser.add_argument("--print_batch_metrics", action='store_true', default=False, help="Set to print metrics for every batch.")
args = parser.parse_args()

# Create output directory
pathlib.Path(args.outdir).mkdir(exist_ok=True)

# Load config or create a new one and save it to outdir for reproducibility
cfg = create_config(args)
save_config(cfg, args.outdir)
print(cfg)

dataset = COCODataset(args.annotations, args.imagedir, image_size = (224,224))
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

NUM_CLASSES = dataset.NUM_CLASSES
print("Number of categories: {}".format(NUM_CLASSES))

model = Model(NUM_CLASSES)
assert(model.TARGET_IMAGE_SIZE == model.CONTEXT_IMAGE_SIZE == dataset.image_size), "Image size from the dataset is not compatible with the encoder."

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate) # TODO: check if this is ok or if model parameters should be passed only after they have been initialized from checkpoint
criterion = nn.CrossEntropyLoss() # TODO: implement custom loss for uncertainty gating

# Send to device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize from checkpoint
if cfg.checkpoint is not None:
    print("Initializing from checkpoint {}".format(cfg.checkpoint))
    checkpoint = torch.load(cfg.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    print("No checkpoint was passed.")
    # TODO: weight initialization
    start_epoch = 1

writer = SummaryWriter(log_dir=os.path.join(args.outdir, "runs/{date:%Y-%m-%d_%H%M}".format(date=datetime.datetime.now())))
context_images, target_images, labels = iter(dataloader).next()
writer.add_images("context_image_batch", context_images) # add example context image batch to tensorboard log
writer.add_images("target_image_batch", target_images) # add example target image batch to tensorboard log
writer.add_graph(model, input_to_model=[context_images.to(device), target_images.to(device)]) # add model graph to tensorboard log
accuracy_logger = AccuracyLogger(dataset.idx2label)


## Training
#

for epoch in tqdm(range(start_epoch, args.epochs + 1), position=0, desc="Epochs", leave=True):

    model.train() # set train mode
    accuracy_logger.reset() # reset accuracy logger every epoch

    for i, (context_images, target_images, labels_cpu) in enumerate(tqdm(dataloader, position=1, desc="Batches", leave=True)):
        context_images = context_images.to(device)
        target_images = target_images.to(device)
        labels = labels_cpu.to(device) # keep a copy of labels on cpu to avoid unnecessary transfer back to cpu later

        optimizer.zero_grad()

        output = model(context_images, target_images)
        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()

        # log metrics
        _, predictions = torch.max(output.detach().to("cpu"), 1) # choose idx with maximum score as prediction
        batch_accuracy = sum(predictions == labels_cpu) / cfg.batch_size
        accuracy_logger.update(predictions, labels_cpu)
        batch_loss = loss.item()

        writer.add_scalar("Batch Loss/train", batch_loss, i + (epoch - 1) * len(dataloader))
        writer.add_scalar("Batch Accuracy/train", batch_accuracy, i + (epoch - 1) * len(dataloader))

        if args.print_batch_metrics:
            print("\t Epoch {}, Batch {}: \t Loss: {} \t Accuracy: {}".format(epoch, i, batch_loss, batch_accuracy))


    # log metrics
    writer.add_scalar("Total Accuracy/train", accuracy_logger.accuracy(), epoch * len(dataloader))
    print("\nEpoch {}, Train Accuracy: {}".format(epoch, accuracy_logger.accuracy()))

    print("{0:20} {1:10}".format("Class", "Accuracy")) # header
    for name, acc in accuracy_logger.named_class_accuarcies().items():
        writer.add_scalar("Class Accuracies/train/{}".format(name), acc, epoch * len(dataloader))
        print("{0:20} {1:10.4f}".format(name, acc))

    # save checkpoint
    if epoch % args.save_frequency == 0:
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, args.outdir + "/checkpoint_{}.tar".format(epoch))
        print("Checkpoint saved.")

    # save training accuracies
    accuracy_logger.save(args.outdir, name="train_accuracies_epoch_{}".format(epoch))
    
    # evaluation on test data
    if cfg.test_annotations is not None and cfg.test_imagedir is not None and epoch % args.test_frequency == 0:
        print("Starting evaluation on test data.")
        test_accuracy = test(model, cfg.test_annotations, cfg.test_imagedir, image_size=dataset.image_size, output_dir=args.outdir, epoch=epoch)

        writer.add_scalar("Total Accuracy/test", test_accuracy.accuracy(), epoch * len(dataloader))
        for name, acc in test_accuracy.named_class_accuarcies().items():
            writer.add_scalar("Class Accuracies/test/{}".format(name), acc, epoch * len(dataloader))

        writer.add_hparams({"learning_rate": cfg.learning_rate}, metric_dict={"hparam/accuracy": test_accuracy.accuracy()})
        
writer.close()