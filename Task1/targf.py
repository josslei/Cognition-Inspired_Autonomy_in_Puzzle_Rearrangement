import functools
import numpy as np
from tqdm import trange

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import knn_graph

from typing import Tuple, List
import copy
import json

from network.score_nets import ScoreNetGNN
from network.sde import marginal_prob_std, diffusion_coeff
from visualization.read_kilogram import read_kilogram
from visualization.visualize_tangrams import draw_tangrams

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TarGF_Tangram_Ball:
    """ Wraps the DSM (Denoising Score-Matching) model's training & inference
    """
    def __init__(self, sigma, path_kilogram_dataset,
                 num_objs=7, is_json=False, batch_size=1016,
                 num_epochs=10000, learning_rate=0.0002, betas=(0.5, 0.999),
                 device=DEVICE) -> None:
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device=device)
        self.num_objs = num_objs

        # Parameters for training
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.betas = betas

        self.path_kilogram_dataset = path_kilogram_dataset
        self.is_json = is_json
        self.batch_size = batch_size
        self.device = device

    def train(self) -> None:
        # Prepare dataset
        # All data are loaded into RAM (of self.device) here
        print('Loading dataset...')
        _dataset = Dataset_KILOGRAM(self.path_kilogram_dataset, self.is_json, self.device)
        dataloader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=True)
        print('Done.')

        # Create model and optimizer
        print('Creating model and optimizer...')
        self.score_net = ScoreNetGNN(marginal_prob_std_func=self.marginal_prob_std_fn,
                                     num_classes=self.num_objs,
                                     device=self.device)
        self.score_net.to(self.device)
        self.optimizer = optim.Adam(self.score_net.parameters(), lr=self.learning_rate, betas=self.betas)
        print('Done.')

        # Training loop
        all_losses = []
        for epoch in trange(self.num_epochs):
            avg_loss = 0.0
            num_items = 0
            # Iterate batches
            for i, data in enumerate(dataloader):
                omega = data[0].view(self.batch_size * self.num_objs, -1)
                edge = data[1].view(2, -1)
                loss = self.__loss_fn(self.score_net, omega, edge)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                all_losses += [loss.item()]
                avg_loss += loss.item() * self.batch_size
                num_items += self.batch_size
            #print('Average Loss: {:5f}'.format(avg_loss / num_items))
            # TODO: Evaluation
            # Save model
            if epoch % 1 == 0:
                import os
                os.system('mkdir -p ./logs/')
                torch.save(self.score_net.state_dict(), f'./logs/score_net_epoch_{epoch}.pt')
                with open('./logs/training_log_losses.txt', 'w') as fp:
                    for loss in all_losses:
                        fp.write(f'{loss}\n')
            pass
        pass

    def __loss_fn(self, model, omega, edge_index, eps=1e-5):
        """The loss function for training score-based generative models.

        Parameters:
            model: A PyTorch model instance that represents a time-dependent score-based model.
            omega: A batch of training data. Shape = (batch_size*num_objs, 3)
            edge_index: A batch of fc graphs. Shape = (2, batch_size*num_objs*(num_objs-1))
            marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
            eps: A tolerance value for numerical stability.
        """
        omega = omega.to(self.device)
        # -> (batch_size * num_objs, 3)
        edge_index = copy.deepcopy(edge_index.to(self.device))

        random_t = torch.rand(self.batch_size, device=self.device) * (1. - eps) + eps
        random_t = random_t.unsqueeze(-1)
        # -> (batch_size, 1)

        #z_position = (torch.randn_like(omega[:,:2]) - 0.5) * 2*omega[:, :2].max()
        ## -> (batch_size * num_objs, 2), interval: [-omega[:,:2].max(), omega[:,:2].max())
        #z_orientation = (torch.randn_like(omega[:,2:]) - 0.5) * 2*np.pi
        ## -> (batch_size * num_objs, 2), interval: [-pi, pi)
        #z = torch.cat([z_position, z_orientation], dim=-1)
        z = torch.randn_like(omega)
        # -> (batch_size * num_objs, 3)

        std = self.marginal_prob_std_fn(random_t)   # -> (batch_size, 1) (shape = random_t.shape)
        std = std.repeat(1, self.num_objs)          # -> (batch_size, num_objs)
        std = std.view(-1, 1)
        # -> (batch_size * num_objs, 1)

        perturbed_omega = copy.deepcopy(omega)  # Can't use torch.clone() because it syncs gradients
        perturbed_omega += z * std  # (batch_size * num_objs, 3) * (batch_size * num_objs, 1)
        # -> (batch_size * num_objs, 3)

        score = model(perturbed_omega, edge_index, random_t, self.num_objs)
        # -> (batch_size * num_objs, 3)

        #loss = torch.mean(torch.sum(((score * std + z)**2).view(self.batch_size, -1), dim=-1))
        loss = (score * std + z)**2             # -> (batch_size * num_objs, 3)
        loss = loss.view(self.batch_size, -1)   # -> (batch_size, 3 * num_objs)
        loss = torch.mean(torch.sum(loss, dim=-1))
        # -> () 0-dimension scalar
        return loss


class Dataset_KILOGRAM(Dataset):
    def __init__(self, path_kilogram_dataset, is_json=False, device=DEVICE) -> None:
        super().__init__()
        if is_json:
            with open(path_kilogram_dataset, 'r') as fp:
                tangram_data = json.load(fp)
        else:
            tangram_data = read_kilogram(path_dataset=path_kilogram_dataset, subset='full')
        # TODO: turn vertices to position and orientation

        self.dataset_omega: List[torch.Tensor] = []
        self.dataset_edge: List[torch.Tensor] = []
        for tk in tangram_data.keys():
            omega = []
            for pk in tangram_data[tk]['positions'].keys():
                o = list(tangram_data[tk]['positions'][pk])
                o += [tangram_data[tk]['orientations'][pk]]
                omega += [o]
            _omega: torch.Tensor = torch.tensor(omega, device=device)
            self.dataset_omega += [_omega]
        # Normalize and create graph
        self.DATA_MAX = 2 + 7 * np.sqrt(2) + np.sqrt(5)
        for i, _ in enumerate(self.dataset_omega):
            # Normalize positions and orientations
            for piece_id, piece in enumerate(self.dataset_omega[i]):
                self.dataset_omega[i][piece_id][0] = piece[0] / self.DATA_MAX
                self.dataset_omega[i][piece_id][1] = piece[1] / self.DATA_MAX
                self.dataset_omega[i][piece_id][2] = piece[2] / np.pi
            # Create graph
            edge: torch.Tensor = knn_graph(self.dataset_omega[i], 7 - 1, loop=False).to(device)
            self.dataset_edge += [edge]
        pass

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Omega & the edge_index.
                Shape of omega: (7, 3)
                Shape of edge_index: (2, 7*6)
        """
        return self.dataset_omega[index], self.dataset_edge[index]

    def __len__(self) -> int:
        return len(self.dataset_omega)
