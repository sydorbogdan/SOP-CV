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
from dataloaders.base_dataset import SOPBaseDataset
import torchmetrics
import torch
import torch.nn.functional as F

from embeder.embeder import Embeder


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


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class EmbederPl(pl.LightningModule):
    def __init__(self, root_dir="./data/Stanford_Online_Products/", head="Linear", num_classes=8796):
        super().__init__()
        self.num_classes = num_classes
        self.root_dir = root_dir
        self.batch_size = 256
        self.head_name = head

        if self.head_name == "Linear":
            self.classification_head = torch.nn.Linear(512, num_classes)
        elif self.head_name == "arcFace":
            self.classification_head = ArcMarginProduct(512, num_classes)

        print(self.classification_head)

        self.embeder = Embeder()

        self.loss_accumulator = Accumulator()
        self.accuracy = torchmetrics.Accuracy(self.num_classes)

    def forward(self, x):
        return self.embeder(x)

    def training_step(self, batch, batch_idx):
        img, class_id, super_class_id, img_id = batch
        label = super_class_id.reshape(super_class_id.shape[0]).long()

        feature = self.embeder(img)
        if self.head_name == "Linear":
            output = self.classification_head(feature)
        elif self.head_name == "arcFace":
            output = self.classification_head(feature, label)

        loss = torch.nn.functional.cross_entropy(input=output, target=label.long())

        pred = torch.argmax(output, dim=1)

        self.loss_accumulator.update(loss.detach().item())
        self.accuracy.update(pred.int().cpu(), label.int().cpu())

        return loss

    def training_epoch_end(self, outputs):
        self.log('train_loss', self.loss_accumulator.compute())
        self.loss_accumulator.clear()

        self.log('train_accuracy', self.accuracy.compute())
        self.accuracy = torchmetrics.Accuracy(self.num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "train_accuracy"}

    def train_dataloader(self):
        return DataLoader(SOPBaseDataset(root_dir=self.root_dir, mode="train"), batch_size=self.batch_size,
                          shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(SOPBaseDataset(root_dir=self.root_dir, mode="val"), batch_size=self.batch_size,
                          num_workers=8)

    def test_dataloader(self):
        return DataLoader(SOPBaseDataset(root_dir=self.root_dir, mode="test"), batch_size=self.batch_size,
                          num_workers=8)


if __name__ == "__main__":
    # Init our model
    model = EmbederPl(root_dir="/home/bohdan/metrics_learning/data/Stanford_Online_Products", head="Linear",
                      num_classes=8796)

    experiment_name = "heh"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/True_SoftMax_all_classes/", save_top_k=2, monitor="train_loss",
        filename=f"{experiment_name}" + "-{epoch:02d}-{train_loss:04f}_{train_accuracy:04f}"
    )

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=50, progress_bar_refresh_rate=20, accelerator="gpu",
                         callbacks=[checkpoint_callback])

    trainer.fit(model)
