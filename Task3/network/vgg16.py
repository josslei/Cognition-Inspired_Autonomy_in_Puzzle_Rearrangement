import torch
import torch.nn as nn
import torchvision


class VGG16(nn.Module):
    """ VGG16 backbone.
    Suggested input size: (224, 224).
    """

    def __init__(self, config: dict, embed_dim: int = 32):
        super().__init__()
        # Expand config
        batch_norm: bool = config['cnn_backbone']['batch_norm']
        pretrained: bool = config['cnn_backbone']['pretrained'] if 'pretrained' in config['cnn_backbone'].keys() else False

        _vgg16: nn.Module
        if batch_norm:
            _vgg16 = torchvision.models.vgg16_bn(pretrained=pretrained)
        else:
            _vgg16 = torchvision.models.vgg16(pretrained=pretrained)

        self.features: nn.Module = _vgg16.features
        self.avgpool: nn.Module = _vgg16.avgpool

        self.embedding = nn.Sequential(
            nn.Linear(25088, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return x
