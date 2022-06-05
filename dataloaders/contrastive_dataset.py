from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import torch
import albumentations as A
import albumentations.pytorch
import random


class SOPContrastiveDataset(Dataset):
    def __init__(self, root_dir, mode="train", augment=True):
        self.root_dir = root_dir

        if mode in ["train", "val"]:
            annotations_path = "Ebay_train.txt"
        elif mode == "test":
            annotations_path = "Ebay_test.txt"
        else:
            raise Exception(f"Invalid dataset mode {mode}")

        self.images_db = {}
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

        self.length = len(lines) - 1

        for line in lines:
            line = line.split()
            if line[1] in self.images_db.keys():
                self.images_db[line[1]] += [{
                    "class_id": int(line[1]),
                    "super_class_id": int(line[2]),
                    "path": os.path.join(self.root_dir, line[3]),
                    "img_id": int(line[0])
                }]
            else:
                self.images_db[line[1]] = [{
                    "class_id": int(line[1]),
                    "super_class_id": int(line[2]),
                    "path": os.path.join(self.root_dir, line[3]),
                    "img_id": int(line[0])
                }]

            self.classes.add(int(line[1]))
            self.super_classes.add(int(line[2]))

            self.id_to_class[str(line[0])] = {
                "class_id": int(line[1]),
                "super_class_id": int(line[2]),
            }

        self.super_classes = list(self.super_classes)
        self.classes = list(self.classes)

        self.num_classes = len(self.classes)
        self.num_super_classes = len(self.super_classes)

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

    def load_image(self, img_meta):
        img = cv2.imread(img_meta["path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))

        img = self.transform(image=img)["image"]
        return img

    def __getitem__(self, idx):

        first_image_class = random.choice(self.classes)
        first_image_meta = random.choice(self.images_db[str(first_image_class)])

        should_get_same_class = random.randint(0, 1)
        if not should_get_same_class:
            while True:
                second_image_class = random.choice(self.classes)
                if second_image_class != first_image_class:
                    break
            second_image_meta = random.choice(self.images_db[str(second_image_class)])
        else:
            if len(self.images_db[str(first_image_class)]) == 1:
                second_image_meta = first_image_meta
            else:
                while True:
                    second_image_meta = random.choice(self.images_db[str(first_image_class)])
                    if second_image_meta["img_id"] != first_image_meta["img_id"]:
                        break

        first_img = self.load_image(first_image_meta)
        second_img = self.load_image(second_image_meta)

        return first_img, second_img, should_get_same_class

    def __len__(self):
        return self.length

    def get_label(self, img_id):
        return self.id_to_class[img_id]


if __name__ == "__main__":
    d = SOPBaseDataset("./../data/Stanford_Online_Products/")

    print(d.num_classes)
    print(d.num_super_classes)
