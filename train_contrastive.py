# from __future__ import print_function
# from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloaders.contrastive_dataset import SOPContrastiveDataset
import torchmetrics
import torch
import torch.nn.functional as F

from embeder.embeder import Embeder


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class Accumulator:
    def __init__(self):
        self.accumulator = 0
        self.counter = 0

    def update(self, value):
        self.accumulator += value
        self.counter += 1

    def compute(self):
        return self.accumulator / self.counter

    def clear(self):
        self.accumulator = 0
        self.counter = 0


class ContrastiveEmbederPl(pl.LightningModule):
    def __init__(self, root_dir="./data/Stanford_Online_Products/", num_classes=8796):
        super().__init__()
        self.num_classes = num_classes
        self.root_dir = root_dir
        self.batch_size = 128

        self.classification_head = torch.nn.Linear(512, num_classes)

        self.embeder = Embeder()

        self.loss = ContrastiveLoss()

        self.loss_accumulator = Accumulator()
        self.accuracy = torchmetrics.Accuracy(self.num_classes)

    def forward(self, x):
        return self.embeder(x)

    def training_step(self, batch, batch_idx):
        img1, img2, same_class = batch

        feature1 = self.embeder(img1)
        feature2 = self.embeder(img2)

        loss = self.loss(feature1, feature2, same_class)

        self.loss_accumulator.update(loss.detach().item())

        return loss

    def training_epoch_end(self, outputs):
        self.log('train_loss', self.loss_accumulator.compute())
        self.loss_accumulator.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "train_loss"}

    def train_dataloader(self):
        return DataLoader(SOPContrastiveDataset(root_dir=self.root_dir, mode="train", augment=True), batch_size=self.batch_size,
                          shuffle=True, num_workers=8)


if __name__ == "__main__":
    # Init our model
    model = ContrastiveEmbederPl(root_dir="data/Stanford_Online_Products",
                                 num_classes=8796)

    experiment_name = "heh"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/Contrastive_loss/", save_top_k=2, monitor="train_loss",
        filename="{epoch:02d}-{train_loss:04f}_{train_accuracy:04f}")

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=50, progress_bar_refresh_rate=1, accelerator="cpu",
                         callbacks=[checkpoint_callback])

    trainer.fit(model)
