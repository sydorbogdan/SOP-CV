from annoy import AnnoyIndex

from dataloaders.base_dataset import SOPBaseDataset
import cv2


u = AnnoyIndex(512, 'angular')
u.load('Indexes/contrastive.ann')

test_dataset = SOPBaseDataset(root_dir="data/Stanford_Online_Products", mode="test")

images = [0 for i in range(test_dataset.num_super_classes)]
counter = 0

for element in test_dataset:
    if images[int(element[2].numpy()[0]) - 1] != 0:
        images[int(element[2].numpy()[0]) - 1] = element[0][None, :, :, :]
        counter += 1
    else:
        continue

    if counter >= test_dataset.num_super_classes:
        break


for c in range(len(images)):
    cv2.imshow(f"class-{c} | instance {i}")
