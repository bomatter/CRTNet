import os
import json
import torch

class AccuracyLogger():
    """
    Logs per-class accuracies and total accuracy.
    """

    def __init__(self, idx2label_dict):
        """
        Parameters:
            idx2label_dict: dictionary mapping class index to class name
        """
        self.idx2label = idx2label_dict
        self.NUM_CLASSES = len(idx2label_dict)

        self.confusionmatrix = torch.zeros(self.NUM_CLASSES, self.NUM_CLASSES) # confusionmatrix[i,j] holds the number of samples with true label i that were classified as j.

    @classmethod
    def from_state_dict(cls, state_dict):
        """
        Initializes an AccuracyLogger from a state_dict.
        """
        ret = cls(state_dict["idx2label_dict"])
        ret.confusionmatrix = state_dict["confusionmatrix"]
        return ret

    def reset(self):
        """
        Resets the logged statistics to 0.
        """
        self.confusionmatrix *= 0

    def update(self, predictions, groundtruth):
        """
        Parameters:
            predictions: model predictions
            groundtruth: groundtruth class indexes corresponding to the predictions.
        """
        assert(len(predictions) == len(groundtruth)), "Predictions and groundtruth should be of the same length."

        for pred, gt in zip(predictions, groundtruth):
            self.confusionmatrix[gt, pred] += 1

    def accuracy(self):
        """
        Returns the total accuracy (first averaged within class and then across classes).
        """
        return (self.confusionmatrix.diag()/(self.confusionmatrix.sum(dim=1) + 1e-7)).mean().item()

    def class_accuracies(self):
        """
        returns a dictionary mapping class ind to accuracy.
        """
        return {idx: acc.item() for idx, acc in enumerate(self.confusionmatrix.diag()/(self.confusionmatrix.sum(dim=1) + 1e-7))}

    def named_class_accuarcies(self):
        """
        returns a dictionary mapping class name to accuracy.
        """
        return {self.idx2label[idx]: acc.item() for idx, acc in enumerate(self.confusionmatrix.diag()/(self.confusionmatrix.sum(dim=1) + 1e-7))}

    def normalized_confusionmatrix(self):
        """
        returns the normalized confusion matrix (as tensor): confusionmatrix[i,j] holds the percentage of classes with real label i that were classified as class j 
        """

        return (self.confusionmatrix.transpose(0,1) / (self.confusionmatrix.sum(dim=1) + 1e-7)).transpose(0,1)

    def state_dict(self):
        """
        Returns a state dict that can be used to restore the AccuracyLogger's state with the from_state_dict class method.
        """
        return {"idx2label_dict": self.idx2label, "confusionmatrix": self.confusionmatrix}

    def save(self, savedir, name="accuracies"):
        """
        Saves logged information to "savedir/name.json".
        """
        savedict = {"total_accuracy": self.accuracy(),
                    "class_accuracies": self.class_accuracies(),
                    "named_class_accuracies": self.named_class_accuarcies(),
                    "confusionmatrix": self.confusionmatrix.tolist(),
                    "normalized_confusionmatrix": self.normalized_confusionmatrix().tolist()}
        with open(os.path.join(savedir, name +".json"), "w") as f:
            json.dump(savedict, f)



class IndividualScoreLogger():
    """
    Keeps track of individual scores: stores for every annotation what the predicted label was and if it was correct.
    The log field holds a list, where each entry is itself a list consisting of [annotation_id, correct (bool), prediction, prediction label name]
    """

    def __init__(self, idx2label_dict):
        self.idx2label = idx2label_dict
        self.NUM_CLASSES = len(idx2label_dict)
        self.log = []

    def update(self, predictions, groundtruth, annotation_ids):
        assert(len(predictions) == len(groundtruth) and len(predictions) == len(annotation_ids)), "Predictions, groundtruth and annotation_ids should be of the same length."

        for pred, gt, annotation_id in zip(predictions, groundtruth, annotation_ids):
            self.log.append([annotation_id.item(), (pred == gt).item(), gt.item(), self.idx2label[gt.item()], pred.item(), self.idx2label[pred.item()]]) # [annotation_id, correct, gt_id, gt_label, pred_id, pred_label]
        
    def save(self, savedir, name="individual_scores"):
        """
        Saves logged information to "savedir/name.json".
        """
        with open(os.path.join(savedir, name +".json"), "w") as f:
            json.dump(self.log, f)