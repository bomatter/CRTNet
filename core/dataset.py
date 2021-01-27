import os
import json
import math

import torch

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, pad, normalize

from PIL import Image
from collections import OrderedDict

class COCODataset(Dataset):
    """
    Dataset to load data in COCO-style format and provide samples corresponding to individual objects.
    A sample consists of a target image (cropped to the objects bounding box), a context image (entire image) and a label in [0,num_classes].
    """

    def __init__(self, annotations_file, image_dir, image_size, idx2label=None, normalize_means=None, normalize_stds=None):
        """
        Args:
            annotations_file: path to COCO-style annotation file (.json)
            image_dir: path to the image folder
            image_size: desired size of the sample images, either a tuple (w,h) or an int if w=h
            idx2label: If a particular mapping between index and label is desired. Format: {idx: "labelname"}.
            normalize_means: List of means for each channel. Set None to disable normalization.
            normalize_stds: List of standard deviations for each channel. Set None to disable normalization.
        """
        
        self.image_dir = image_dir
        self.image_size = image_size if type(image_size) == tuple else (image_size, image_size)

        with open(annotations_file) as f:
            self.coco_dict = json.load(f, object_pairs_hook=OrderedDict)

        self.annotations = self.coco_dict["annotations"]
        
        self.id2file = {}
        for i in self.coco_dict["images"]:
            self.id2file[i["id"]] = os.path.join(image_dir, i["file_name"])

        self.id2label = {} # maps label id to label name
        self.id2idx = {} # maps label id to index in 1-hot encoding
        if idx2label is None:
            self.idx2label = {} # maps index in 1-hot encoding to label name
            for idx, i in enumerate(self.coco_dict["categories"]):
                self.id2label[i["id"]] = i["name"]    
                self.idx2label[idx] = i["name"]
                self.id2idx[i["id"]] = idx
        else:
            assert(len(self.coco_dict["categories"]) == len(idx2label)), "Number of categorires in the annotation file does not agree with the number of categories in the custom idx2label mapping."
            
            self.idx2label = idx2label # maps index in 1-hot encoding to label name
            label2idx = {label: idx for idx, label in self.idx2label.items()}
            for i in self.coco_dict["categories"]:
                self.id2label[i["id"]] = i["name"]
                self.id2idx[i["id"]] = label2idx[i["name"]]
        
        self.NUM_CLASSES = len(self.id2label)

        if normalize_means is not None and normalize_stds is not None:
            self.normalize_means = normalize_means
            self.normalize_stds = normalize_stds
            self.normalize = True
        else:
            self.normalize = False

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        # load image
        image = Image.open(self.id2file[annotation["image_id"]])
        image = image.convert("RGB")
        
        # crop to bounding box for target image
        xmin, ymin, w, h = annotation["bbox"]
        target_image = image.crop((int(xmin), int(ymin), int(xmin + w), int(ymin + h)))

        # resize
        image = image.resize(self.image_size)
        target_image = target_image.resize(self.image_size)

        # convert to torch tensor
        image = to_tensor(image)
        target_image = to_tensor(target_image)

        # normalize
        if self.normalize:
            image = normalize(image, self.normalize_means, self.normalize_stds)
            target_image = normalize(target_image, self.normalize_means, self.normalize_stds)

        label = self.id2idx[annotation["category_id"]]

        return image, target_image, label


class COCODatasetWithID(COCODataset):
    """
    Provides the same functionality as COCODataset but in addition the id of the annotation is also returned.
    This can be useful for in-depth analysis of the results.
    """

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        annotation_id = self.annotations[idx]["id"]
        return (*sample, annotation_id)