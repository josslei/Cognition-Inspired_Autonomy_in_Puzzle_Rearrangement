import torch
import torch.nn as nn
import torchvision


class VGG16(nn.Module):
    """ VGG16 backbone.
    Suggested input size: (448, 448) or (224, 224).
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()

        # pretrained_vgg16: nn.Module = torchvision.models.vgg16_bn(pretrained=True)
        pretrained_vgg16: nn.Module = torchvision.models.vgg16(pretrained=True)

        self.features: nn.Module = pretrained_vgg16.features
        self.avgpool: nn.Module = pretrained_vgg16.avgpool

        self.embedding = nn.Sequential(
            nn.Linear(25088, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.embedding(x)
        return x
