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

class ScoreNetGNN(nn.Module):
    """ GNN ScoreNet (basic form)

    Ref: TarGF's ScoreNetGNN for ball (re)arrangement. [TarGF/networks/score_nets.py:107]

    """
    def __init__(self, marginal_prob_std_func, num_classes=7, device=DEVICE, hidden_dim=64, embed_dim=32):
        super().__init__()

        self.device = device
        self.num_classes = num_classes

        # x-feature
        self.init_lin = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # t-feature
        self.embed_t = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )

        ##
        ## Here we let conv(input_conv) = EdgeConv(input_conv),
        ## and EdgeConv uses these MLPs as their :math:`h_{\mathbf{\Theta}}`.
        ##
        ## If the shape of input_conv is (*, dim),
        ## the shape of MLP's input should be (*, dim * 2)
        ##
        ## Reason:
        ##  - In the paper, the finally adopted edge function is defined as:
        ##  :math:`h_{\mathbf{\Theta}}(\bm{x}_i, \bm{x}_j) = \bar{h}_{\mathbf{\Theta}}(\bm{x}_i, \bm{x}_j - \bm{x}_i)`.
        ##  - Specifically, in the code, the input to the MLP (or any other :math:`h_{\mathbf(\Theta)}`)
        ##  is defined as `torch.cat([x_i, x_j - x_i], dim=-1)`
        ##
        # conv1
        # input_conv.shape = (batch_size * num_objs, hidden_dim)
        # input_mlp.shape = (batch_size * num_objs, hidden_dim * 2)
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = EdgeConv(self.mlp1)
        # -> (batch_size * num_objs, hidden_dim)

        # conv2
        # input_conv.shape = (batch_size * num_objs, hidden_dim + embed_dim)
        # input_mlp.shape = (batch_size * num_objs, (hidden_dim + embed_dim) * 2)
        self.mlp2 = nn.Sequential(
            nn.Linear((hidden_dim + embed_dim) * 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = EdgeConv(self.mlp2)
        # -> (batch_size * num_objs, hidden_dim)

        # conv3
        # input_conv.shape = (batch_size * num_objs, hidden_dim + embed_dim)
        # input_mlp.shape = (batch_size * num_objs, (hidden_dim + embed_dim) * 2)
        self.mlp3 = nn.Sequential(
            nn.Linear((hidden_dim + embed_dim) * 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 3),
        )
        self.conv3 = EdgeConv(self.mlp3)
        # -> (batch_size * num_objs, 3)

        self.marginal_prob_std = marginal_prob_std_func
        pass

    def forward(self, omega, edge_index, t, num_objs=None):
        """ Network forward

        Args:
            omega (torch.Tensor): state of tangram pieces, shape = (batch_size*num_objs, 3)
            edge_index (torch.Tensor): graph that connects pieces, shape = (2, batch_size*num_objs*(num_objs-1))
            t (torch.Tensor): time instant(s), shape = (batch_size, 1)
            num_objs (int): number of objects, in this case is the number of tangram pieces
        """
        if num_objs == None:
            num_objs = self.num_classes
        # Extract initial feature
        init_feature = self.init_lin(omega)
        # -> (batch_size * num_objs, hidden_dim)

        # Get t-feature
        batch_size = t.shape[0]
        x_sigma = F.relu(self.embed_t(t.squeeze(1)))   # -> (batch_size, embed_dim)
        x_sigma = x_sigma.squeeze(1).repeat(1, num_objs, 1).view(batch_size * num_objs, -1)
        # -> (batch_size * num_objs, embed_dim)

        # Start message passing from init-feature
        x = F.relu(self.conv1(init_feature, edge_index))
        x = torch.cat([x, x_sigma], dim=-1)
        x = F.relu(self.conv2(x, edge_index))
        x = torch.cat([x, x_sigma], dim=-1)
        x = self.conv3(x, edge_index)
        # -> (batch_size * num_objs, 3)

        # Normalization
        t_reshaped = t.repeat(1, num_objs).view(batch_size * num_objs, -1)
        x = x / (self.marginal_prob_std(t_reshaped) + 1e-7)
        return x