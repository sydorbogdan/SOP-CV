from annoy import AnnoyIndex
from dataloaders.base_dataset import SOPBaseDataset
import tqdm

from train import EmbederPl
from train_contrastive import ContrastiveEmbederPl


def index_dataset(_model, _dataset):
    emb_length = model(d[0][0][None, :, :, :])[0]

    t = AnnoyIndex(emb_length.shape[-1], 'angular')

    for idx, inst in tqdm.tqdm(enumerate(_dataset)):
        img, class_id, super_class_id, img_id = inst

        emb = _model(img[None, :, :, :])[0]

        t.add_item(i=img_id, vector=emb)

    t.build(n_trees=500)

    t.save('pretrained.ann')

    return t


if __name__ == "__main__":
    d = SOPBaseDataset("./data/Stanford_Online_Products/", mode="train", augment=False)
    #
    # print(f"len {len(d)}")

    # model = EmbederPl(root_dir="/home/bohdan/metrics_learning/data/Stanford_Online_Products", head="Linear",
    #                     num_classes=8796)
    #
    # model.load_from_checkpoint(
    #     "/home/bohdan/Documents/UCU/3/CV/hehe/checkpoints/True_SoftMax_all_classes/heh-epoch=41-train_loss=0.002539_train_accuracy=0.999538.ckpt")

    model = ContrastiveEmbederPl(root_dir="/home/bohdan/metrics_learning/data/Stanford_Online_Products",
                                 num_classes=8796)

    # model.load_from_checkpoint(
    #     "/home/bohdan/Documents/UCU/3/CV/hehe/checkpoints/Contrastive_loss/epoch=31-train_loss=1.005310_train_accuracy=0.000000.ckpt")

    # t = index_dataset(model, [d[i] for i in range(100)])
    t = index_dataset(model, d)

    u = AnnoyIndex(512, 'angular')
    u.load('arcFace.ann')  # super fast, will just mmap the file

    # print(u.get_nns_by_vector(model(d[0][0]).detach().numpy(), 10))

    # print(torch.flatten(resnet18_embedder(d[0][0].permute(2, 1, 0)[None, :, :, :]), 1).shape)

    # resnet18_embedder()
