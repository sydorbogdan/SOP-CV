from annoy import AnnoyIndex
import tqdm
import torch
import numpy as np

from dataloaders.base_dataset import SOPBaseDataset
from train import EmbederPl
from train_contrastive import ContrastiveEmbederPl


class Accuracy:
    """Class for accuracy calculation"""

    def __init__(self):
        self.correct_cases = 0
        self.total_cases = 0

    def update(self, _is_equal):
        if _is_equal:
            self.correct_cases += 1
            self.total_cases += 1
        else:
            self.total_cases += 1

    def compute(self):
        return self.correct_cases / self.total_cases

    def clear(self):
        self.correct_cases = 0
        self.total_cases = 0


class F1:
    """Class for precision, recall and f1 calculation"""

    def __init__(self, num_classes):
        # for each class: tp, tn, fp, fn
        self.num_classes = num_classes
        self.metrics_table = np.zeros((num_classes, 4))

    def update(self, pred, target):
        if pred == target:
            self.metrics_table[pred][0] += 1  # tp

        elif pred != target:
            self.metrics_table[pred][2] += 1  # fp
            self.metrics_table[target.astype(np.int64)][3] += 1  # fn

    def recall(self):
        return self.metrics_table[:, 0] / (
                self.metrics_table[:, 0] + self.metrics_table[:, 3] + 1e-7)

    def precision(self):
        return self.metrics_table[:, 0] / (
                self.metrics_table[:, 0] + self.metrics_table[:, 2] + 1e-7)

    def f1(self):
        return (2 * self.precision() * self.recall()) / (self.precision() + self.recall() + 1e-7)


def get_class(index, _emb, _dataset):
    """Function which predicts class. It uses top 50 retrievals and returns the most popular class"""

    n_closest = index.get_nns_by_vector(torch.transpose(_emb, 0, 1), 50, search_k=-1, include_distances=False)
    classes_counter = np.zeros(_dataset.num_classes)
    super_classes_counter = np.zeros(_dataset.num_super_classes)

    for closest in n_closest:
        classes = _dataset.get_label(str(closest))
        _class_id = classes["class_id"]
        _super_class_id = classes["super_class_id"]

        classes_counter[_class_id - 1] += 1
        super_classes_counter[_super_class_id - 1] += 1

    pred_class_logits = classes_counter.argmax()
    pred_super_class_logits = super_classes_counter.argmax()

    return pred_class_logits, pred_super_class_logits


def test():
    u = AnnoyIndex(512, 'angular')
    u.load('Indexes/softmax.ann')

    dataset = SOPBaseDataset(root_dir="data/Stanford_Online_Products", mode="train")
    val_dataset = SOPBaseDataset(root_dir="data/Stanford_Online_Products", mode="val")

    model = EmbederPl(root_dir="/home/bohdan/metrics_learning/data/Stanford_Online_Products", head="arcFace",
                      num_classes=8796)

    model.load_from_checkpoint(
    #     "/home/bohdan/Documents/UCU/3/CV/hehe/checkpoints/SoftMax_all_classes/heh-epoch=39-train_loss=0.149368_train_accuracy=0.985076.ckpt")
        "/home/bohdan/Documents/UCU/3/CV/hehe/checkpoints/True_SoftMax_all_classes/heh-epoch=41-train_loss=0.002539_train_accuracy=0.999538.ckpt")


    # model = ContrastiveEmbederPl(root_dir="/home/bohdan/metrics_learning/data/Stanford_Online_Products",
    #                              num_classes=8796)

    # model.load_from_checkpoint(
    #     "/home/bohdan/Documents/UCU/3/CV/hehe/checkpoints/Contrastive_loss/epoch=31-train_loss=1.005310_train_accuracy=0.000000.ckpt")

    class_f1 = F1(num_classes=max(list(set(dataset.classes).union(set(val_dataset.classes)))))
    super_class_f1 = F1(num_classes=max(list(set(dataset.super_classes).union(set(val_dataset.super_classes)))))

    class_acc = Accuracy()
    super_class_acc = Accuracy()

    counter = 0
    for i in tqdm.tqdm(dataset):
        img, class_id, super_class_id, img_id = i

        counter += 1
        if counter > 500:
            break

        emb = model(img[None, :, :, :])

        pred_class_id, pred_super_class_id = get_class(u, emb, dataset)

        class_f1.update(target=class_id.numpy()[0] - 1, pred=pred_class_id)
        super_class_f1.update(target=super_class_id.numpy()[0] - 1, pred=pred_super_class_id)
        class_acc.update((class_id.numpy()[0] - 1) == pred_class_id)
        super_class_acc.update((super_class_id.numpy()[0] - 1) == pred_super_class_id)

    print(f"Class precision: {class_f1.precision().mean()}")
    print(f"Class recall: {class_f1.recall().mean()}")
    print(f"Class f1: {class_f1.f1().mean()}")
    print(f"Class accuracy: {class_acc.compute()}")
    print(f"Summary: {class_f1.metrics_table.sum(axis=0)}")
    # print(f"Class accuracy: {class_metrics.accuracy().mean()}")
    print("\n")
    print(f"Super class precision: {super_class_f1.precision().mean()}")
    print(f"Super class recall: {super_class_f1.recall().mean()}")
    print(f"Super class f1: {super_class_f1.f1().mean()}")
    print(f"Super class accuracy: {super_class_acc.compute()}")
    print(f"Summary: {super_class_f1.metrics_table.sum(axis=0)}")
    # print(f"Super class accuracy: {class_metrics.accuracy().mean()}")


if __name__ == "__main__":
    test()
