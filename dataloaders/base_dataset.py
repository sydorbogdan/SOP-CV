from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import torch
import albumentations as A
import albumentations.pytorch
import random

random.seed(1010)


class SOPBaseDataset(Dataset):
    def __init__(self, root_dir, mode="train", augment=True):
        self.root_dir = root_dir

        if mode in ["train", "val"]:
            annotations_path = "Ebay_train.txt"
        elif mode == "test":
            annotations_path = "Ebay_test.txt"
        else:
            raise Exception(f"Invalid dataset mode {mode}")

        self.images_db = []
        self.id_to_class = {}
        self.super_classes = set()
        self.classes = set()

        with open(os.path.join(self.root_dir, annotations_path)) as f:
            lines = f.readlines()
            lines = lines[1:]
            random.shuffle(lines)
            if mode == "train":
                lines = lines[: int(0.8 * len(lines))]
            elif mode == "val":
                lines = lines[int(0.8 * len(lines)):]

        for line in lines[1:]:
            line = line.split()
            self.images_db += [{
                "class_id": int(line[1]),
                "super_class_id": int(line[2]),
                "path": os.path.join(self.root_dir, line[3]),
                "img_id": int(line[0])
            }]

            if int(line[1]) == 11264:
                print(f"heheheheheheh {int(line[1])}")

            self.classes.add(int(line[1]))
            self.super_classes.add(int(line[2]))

            self.id_to_class[str(line[0])] = {
                "class_id": int(line[1]),
                "super_class_id": int(line[2]),
            }

        self.super_classes = list(self.super_classes)
        self.classes = list(self.classes)

        self.num_classes = max(self.classes)
        self.num_super_classes = max(self.super_classes)

        # transformations
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.augmentations.transforms.Normalize(),  # ImageNet normalization
                albumentations.pytorch.transforms.ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.augmentations.transforms.Normalize(),  # ImageNet normalization
                albumentations.pytorch.transforms.ToTensorV2(),
            ])

    def __getitem__(self, idx):
        img = cv2.imread(self.images_db[idx]["path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))

        img = self.transform(image=img)["image"]

        class_id = torch.Tensor([self.images_db[idx]["class_id"]])
        super_class_id = torch.Tensor([self.images_db[idx]["super_class_id"]])
        img_id = self.images_db[idx]["img_id"]

        return img, class_id, super_class_id, img_id

    def __len__(self):
        return len(self.images_db)

    def get_label(self, img_id):
        return self.id_to_class[img_id]


if __name__ == "__main__":
    d = SOPBaseDataset("./../data/Stanford_Online_Products/")

    print(d.num_classes)
    print(d.num_super_classes)
