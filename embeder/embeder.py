import torch
import torchvision.models as models


class Embeder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)

        self.resnet18_embedder = torch.nn.Sequential(
            self.resnet18.conv1,
            self.resnet18.bn1,
            self.resnet18.relu,
            self.resnet18.maxpool,

            self.resnet18.layer1,
            self.resnet18.layer2,
            self.resnet18.layer3,
            self.resnet18.layer4,

            self.resnet18.avgpool
        )

    def forward(self, x):
        # if len(x.shape) == 3:
        #     return torch.flatten(self.resnet18_embedder(x[None, :, :, :]), 1)
        # else:
        #     # print(f"flatten {torch.flatten(self.resnet18_embedder(x)).shape}")

        return torch.flatten(self.resnet18_embedder(x), 1)


if __name__ == "__main__":
    model = Embeder(3)

    dummy_input = torch.randn((3, 3, 128, 128))
    dummy_label = torch.randn((3, 3))

    print(f"{dummy_input.requires_grad=}")
    print(dummy_input.shape)

    embeddings, logits = model(dummy_input)

    print(f"{logits.requires_grad=}")
    # print(f"{dummy_output['logits'][:, None].requires_grad=}")
    # print(f"{dummy_output['logits'][:, None].float().requires_grad=}")

    loss = torch.nn.functional.cross_entropy(logits.float(),
                                             dummy_label.float())

    print(f"{loss.requires_grad=}")
    print(loss)

    # pred = torch.argmax(dummy_output["logits"], dim=1)
    #
    # print(f"{pred.requires_grad=}")
    # print(f"{dummy_output['classification_outputs'].requires_grad=}")
