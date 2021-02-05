import argparse
import pathlib

import torch, torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from core.config import create_config, save_config
from core.dataset import COCODataset, COCODatasetWithID
from core.model import Model
from core.metrics import AccuracyLogger, IndividualScoreLogger


def test(model, annotations_file, image_dir, image_size, output_dir, epoch=None, record_individual_scores=False, print_batch_metrics=False):
    """
    Arguments:
        epoch: If specified, it is used to include the epoch in the output file name.
    """
    pathlib.Path(output_dir).mkdir(exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    testset = COCODatasetWithID(annotations_file, image_dir, image_size, normalize_means=[0.485, 0.456, 0.406], normalize_stds=[0.229, 0.224, 0.225])
    dataloader = DataLoader(testset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)

    if print_batch_metrics:
        criterion = nn.CrossEntropyLoss()

    test_accuracy = AccuracyLogger(testset.idx2label)

    if record_individual_scores:
        individual_scores = IndividualScoreLogger(testset.idx2label)
    
    model.eval() # set eval mode
    with torch.no_grad():
        for i, (context_images, target_images, bbox, labels_cpu, annotation_ids) in enumerate(tqdm(dataloader, desc="Test Batches", leave=True)):
            context_images = context_images.to(device)
            target_images = target_images.to(device)
            bbox = bbox.to(device)
            labels = labels_cpu.to(device) # keep a copy of labels on cpu to avoid unnecessary transfer back to cpu later

            output = model(context_images, target_images, bbox) # output is (batchsize, num_classes) tensor of logits
            _, predictions = torch.max(output.detach().to("cpu"), 1) # choose idx with maximum score as prediction
            test_accuracy.update(predictions, labels_cpu)

            if record_individual_scores:
                individual_scores.update(output.to("cpu"), labels_cpu, annotation_ids)

            # print
            if print_batch_metrics:
                batch_loss = criterion(output, labels).item()
                batch_corr = sum(predictions == labels_cpu) # number of correct predictions
                batch_accuracy = batch_corr # / batch_size # since batchsize is 1

                print("\t Test Batch {}: \t Loss: {} \t Accuracy: {}".format(i, batch_loss, batch_accuracy))
        
    print("\nTotal Test Accuracy: {}".format(test_accuracy.accuracy()))
    print("{0:20} {1:10}".format("Class", "Accuracy")) # header
    for name, acc in test_accuracy.named_class_accuarcies().items():
        print("{0:20} {1:10.4f}".format(name, acc))

    # save accuracies
    if epoch is not None:
        test_accuracy.save(output_dir, name="test_accuracies_epoch_{}".format(epoch))
    else:
        test_accuracy.save(output_dir, name="test_accuracies")

    if record_individual_scores:
        individual_scores.save(output_dir)

    return test_accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("--outdir", type=str, default="evaluation", help="Path to output folder (will be created if it does not exist).")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint.")
    parser.add_argument("--annotations", type=str, help="Path to COCO-style annotations file.")
    parser.add_argument("--imagedir", type=str, help="Path to images folder w.r.t. which filenames are specified in the annotations.")
    
    # TODO: would be nicer to load / infer model parameters from config or checkpoint
    parser.add_argument("--num_classes", type=int, default=33, help="Number of classes.")
    parser.add_argument("--num_decoder_heads", type=int, help="Number of decoder heads.")
    parser.add_argument("--num_decoder_layers", type=int, help="Number of decoder layers.")
    
    parser.add_argument("--image_size", type=tuple, default=(224, 224), help="Input image size the model requires.")
    parser.add_argument('--record_individual_scores', action='store_true', default=False, help="If set, will log for each individual annotion how it was predicted and if the prediction was correct")
    parser.add_argument("--print_batch_metrics", action='store_true', default=False, help="Set to print metrics for every batch.")
    args = parser.parse_args()

    assert(args.checkpoint is not None), "No checkpoint was passed. A checkpoint is required to load the model."

    # TODO: could create config files for reproducibility of evaluation

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Initializing model from checkpoint {}".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    model = Model(args.num_classes, num_decoder_layers=args.num_decoder_layers, num_decoder_heads=args.num_decoder_heads)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    test(model, args.annotations, args.imagedir, args.image_size, args.outdir, record_individual_scores=args.record_individual_scores , print_batch_metrics=args.print_batch_metrics)
