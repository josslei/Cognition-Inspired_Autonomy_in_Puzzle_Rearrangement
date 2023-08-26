import torch
import torch.nn as nn

from typing import List, Tuple


class SimpleCNN(nn.Module):
    """ A simple CNN backbone
    Suggested input size: (224, 224)
    """
    def __init__(self, config: dict, embed_dim: int = 32, num_class: int = 1016):
        super().__init__()
        # Expand config
        batch_norm: bool = config['cnn_backbone']['batch_norm']
        cnn_structure: list = config['cnn_backbone']['cnn_structure']

        # Define model
        _conv2d_list: List[nn.Module] = []
        in_channels: int = 3
        for v in cnn_structure:
            if v == 'M':
                _conv2d_list += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(v) is int:
                _conv2d: nn.Module = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
                if batch_norm:
                    _conv2d_list += [_conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    _conv2d_list += [_conv2d, nn.ReLU(inplace=True)]
                in_channels = v
            else:
                raise Exception('CNN structure definition syntax error.')
        out_channels = in_channels
        self.features: nn.Module = nn.Sequential(*_conv2d_list)
        self.avgpool: nn.Module = nn.AdaptiveAvgPool2d((7, 7))
        self.embedding = nn.Sequential(
            nn.Linear(7 * 7 * out_channels, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.auxiliary = nn.Sequential(
            nn.Linear(7 * 7 * out_channels, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, num_class)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_embedding: torch.Tensor = self.embedding(x)
        x_auxiliary: torch.Tensor = self.auxiliary(x)
        return x_embedding, x_auxiliary