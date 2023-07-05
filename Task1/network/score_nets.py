import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import EdgeConv
from torch.nn.parameter import Parameter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ScoreNetTangram(nn.Module):
    """ ScoreNet tangram (re)arrangement.
    
    Ref: [TarGF/networks/score_nets.py:107]

    """
    def __init__(self, marginal_prob_std_func, num_objs=7, device=DEVICE, hidden_dim=64, embed_dim=32):
        super().__init__()

        self.device = device
        self.num_objs = num_objs

        # x-feature
        self.init_lin = nn.Sequential(
            nn.Linear(self.num_objs * 3, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # t-feature
        self.embed_t = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )

        # mlp1
        # input shape: (batch_size, hidden_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True)
        )
        # -> (batch_size, hidden_dim)

        # mlp2
        # input shape = (batch_size, hidden_dim + embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True)
        )
        # -> (batch_size, hidden_dim)

        # mlp3
        # input shape = (batch_size, hidden_dim + embed_dim)
        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True)
        )
        # -> (batch_size, hidden_dim)

        # mlp4
        # input shape = (batch_size, hidden_dim + embed_dim)
        self.mlp4 = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True)
        )
        # -> (batch_size, hidden_dim)

        # mlp5
        # input shape = (batch_size, hidden_dim + embed_dim)
        self.mlp5 = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True)
        )
        # -> (batch_size, hidden_dim)

        # mlp6
        # input shape = (batch_size, hidden_dim + embed_dim)
        self.mlp6 = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True)
        )
        # -> (batch_size, hidden_dim)

        # mlp7
        # input shape = (batch_size, hidden_dim + embed_dim)
        self.mlp7 = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True)
        )
        # -> (batch_size, hidden_dim)

        # mlp8
        # input shape = (batch_size, hidden_dim + embed_dim)
        self.mlp8 = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True)
        )
        # -> (batch_size, hidden_dim)

        # mlp_end
        # input shape = (batch_size, hidden_dim + embed_dim)
        self.mlp_end = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, self.num_objs * 3),
        )
        # -> (batch_size, 3)

        self.marginal_prob_std = marginal_prob_std_func
        pass

    def forward(self, omega, t, num_objs=None):
        """ Network forward

        Args:
            omega (torch.Tensor): state of tangram pieces, shape = (batch_size * num_objs, 3)
            t (torch.Tensor): time instant(s), shape = (batch_size, 1)
            num_objs (int): number of objects, in this case is the number of tangram pieces
        """
        if num_objs == None:
            num_objs = self.num_objs
        
        batch_size = t.shape[0]
        omega = omega.view(batch_size, num_objs * 3)

        # Extract initial feature
        init_feature = self.init_lin(omega)
        # -> (batch_size, hidden_dim)

        # Get t-feature
        batch_size = t.shape[0]
        x_sigma = F.relu(self.embed_t(t.squeeze(1)))
        # -> (batch_size, embed_dim)

        # Start message passing from init-feature
        x = self.mlp1(init_feature)
        x = torch.cat([x, x_sigma], dim=-1)
        x = self.mlp2(x)
        x = torch.cat([x, x_sigma], dim=-1)
        x = self.mlp3(x)
        x = torch.cat([x, x_sigma], dim=-1)
        x = self.mlp4(x)
        x = torch.cat([x, x_sigma], dim=-1)
        x = self.mlp5(x)
        x = torch.cat([x, x_sigma], dim=-1)
        x = self.mlp6(x)
        x = torch.cat([x, x_sigma], dim=-1)
        x = self.mlp7(x)
        x = torch.cat([x, x_sigma], dim=-1)
        x = self.mlp8(x)
        x = torch.cat([x, x_sigma], dim=-1)
        x = self.mlp_end(x)
        # -> (batch_size, num_obj * 3)

        # Normalization
        x = x / (self.marginal_prob_std(t) + 1e-7)
        return x