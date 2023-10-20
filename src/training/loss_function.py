import torch
from torch.nn import BCELoss
from torchmetrics.classification import BinaryAccuracy

class XorLoss():

    def __init__(self, hp):

        self.loss = BCELoss()
        self.acc_metric = BinaryAccuracy()

        pass

    def loss_fn(self, pred_probs, target_labels):

        loss_batch = self.loss(pred_probs, target_labels)

        accuracy_batch = self.acc_metric(pred_probs, target_labels)

        return loss_batch, accuracy_batch
