import os
import functools
import numpy as np

from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import List, Tuple, Dict, Union
from itertools import chain
import copy
import cv2

import samplers
import evaluator
from network.score_nets import ScoreNetTangram
from network.sde import marginal_prob_std, diffusion_coeff
from dataset import Dataset_KILOGRAM, DataLoaderX
from visualization.visualize_tangrams import draw_tangrams, images_to_video

from network.vgg16 import VGG16
from network.simple_cnn import SimpleCNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TarGF_Tangram:
    """ Wraps the DSM (Denoising Score-Matching) model's training & inference
    """

    def __init__(self,
                 config: dict,
                 sigma: float = 25,
                 num_objs: int = 7,
                 betas: Tuple[float, float] = (0.5, 0.999),
                 device: torch.device = DEVICE) -> None:
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
        self.num_objs = num_objs

        # Parameter for training
        self.betas = betas

        self.device = device

        # All data are loaded into RAM (of self.device) here
        print('Loading dataset...')
        self._dataset: Dataset_KILOGRAM = Dataset_KILOGRAM(config, self.device)
        print('Done.')

    def train(self, config: dict, log_save_dir: str) -> None:
        # Expand config - dataset
        input_image_type: str = config['dataset']['input_image_type']   # 'concrete' or 'segmentation'
        # Expand config - score_net
        num_fc_layers: int = config['score_net']['num_fc_layers']
        hidden_dim: int = config['score_net']['hidden_dim']
        embed_dim: int = config['score_net']['embed_dim']
        score_net_checkpoint: str = config['score_net']['checkpoint'] if 'checkpoint' in config['score_net'].keys() else ''
        # Expand config - cnn_backbone
        cnn_backbone_model: str = config['cnn_backbone']['model']
        cnn_backbone_checkpoint: str = config['cnn_backbone']['checkpoint'] if 'checkpoint' in config['cnn_backbone'].keys() else ''
        # Expand config - training
        learning_rate: float = config['training']['learning_rate']
        num_epochs: int = config['training']['num_epochs']
        batch_size: int = config['training']['batch_size']
        cnn_auxiliary_flag: bool = config['training']['cnn_auxiliary']

        # Create dataloader
        self.dataloader = DataLoaderX(self._dataset, batch_size=batch_size, shuffle=True)

        # Create models and optimizer
        print('Creating models and optimizer...')
        score_net = ScoreNetTangram(marginal_prob_std_func=self.marginal_prob_std_fn,
                                    num_objs=self.num_objs,
                                    device=self.device,
                                    num_fc_layers=num_fc_layers,
                                    hidden_dim=hidden_dim,
                                    embed_dim=embed_dim)
        if score_net_checkpoint != '':
            state_dict_score_net = torch.load(score_net_checkpoint)
            score_net.load_state_dict(state_dict_score_net, strict=False)
        score_net.to(self.device)
        cnn_backbone: nn.Module = eval(cnn_backbone_model)(config=config,
                                                           embed_dim=embed_dim,
                                                           num_class=self._dataset.num_class)
        if cnn_backbone_checkpoint != '':
            state_dict_cnn_backbone = torch.load(cnn_backbone_checkpoint)
            cnn_backbone.load_state_dict(state_dict_cnn_backbone)
        cnn_backbone.to(self.device)
        optimizer = optim.Adam(params=chain(cnn_backbone.parameters(), score_net.parameters()),
                               lr=learning_rate, betas=self.betas)
        print('Done.')

        # Training loop
        avg_losses_per_epoch_score_net: List[float] = []
        avg_losses_per_epoch_cnn_auxiliary: List[float] = []
        score_net_dict_path: str = ''
        cnn_backbone_dict_path: str = ''
        for epoch in trange(num_epochs):
            # Iterate batches
            _avg_loss_score_net: float = 0.0
            _avg_loss_cnn_auxiliary: float = 0.0
            _num_iters: int = 0
            for _, data in enumerate(self.dataloader):
                omega: torch.Tensor = data[0].view(batch_size, self.num_objs * 3)
                concrete_images: torch.Tensor = data[1]
                segmentation_images: torch.Tensor = data[2]
                enhanced_images: torch.Tensor = data[3]
                binarized_images: torch.Tensor = data[4]
                class_labels: torch.Tensor = data[-1].to(self.device)
                input_images: torch.Tensor = eval(input_image_type + '_images') # 'concrete' or 'segmentation'
                loss_score_net: float; loss_cnn_auxiliary: float
                loss_score_net, loss_cnn_auxiliary = self.__train_one_epoch(model=score_net,
                                                                            cnn_backbone=cnn_backbone,
                                                                            omega=omega,
                                                                            input_images=input_images,
                                                                            class_labels=class_labels,
                                                                            optimizer=optimizer,
                                                                            batch_size=batch_size,
                                                                            cnn_auxiliary_flag=cnn_auxiliary_flag)
                _avg_loss_score_net += loss_score_net
                _avg_loss_cnn_auxiliary += loss_cnn_auxiliary
                _num_iters += 1
            avg_losses_per_epoch_score_net += [_avg_loss_score_net / _num_iters]
            avg_losses_per_epoch_cnn_auxiliary += [_avg_loss_cnn_auxiliary / _num_iters]
            # TODO: Evaluation
            # Save model
            if (epoch + 1) % 500 == 0:
                os.system(f'mkdir -p {log_save_dir}')
                if os.path.exists(score_net_dict_path):
                    os.remove(score_net_dict_path)
                if os.path.exists(cnn_backbone_dict_path):
                    os.remove(cnn_backbone_dict_path)
                score_net_dict_path = os.path.join(log_save_dir, f'score_net_epoch_{epoch}.pt')
                cnn_backbone_dict_path = os.path.join(log_save_dir, f'cnn_backbone_epoch_{epoch}.pt')
                torch.save(score_net.state_dict(), score_net_dict_path)
                torch.save(cnn_backbone.state_dict(), cnn_backbone_dict_path)
                # Record loss
                with open(os.path.join(log_save_dir, 'training_log_losses_score_net.txt'), 'w') as fp:
                    for loss in avg_losses_per_epoch_score_net:
                        fp.write(f'{loss}\n')
                if cnn_auxiliary_flag:
                    fp = open(os.path.join(log_save_dir, 'training_log_losses_cnn_auxiliary.txt'), 'w')
                    for loss in avg_losses_per_epoch_cnn_auxiliary:
                        fp.write(f'{loss}\n')
                    fp.close()
            pass
        pass

    def __train_one_epoch(self,
                          model, cnn_backbone,
                          omega, input_images, class_labels,
                          optimizer,
                          batch_size: int,
                          cnn_auxiliary_flag: bool) -> Tuple[float, float]:
        cnn_feature: torch.Tensor; cnn_auxiliary: torch.Tensor
        cnn_feature, cnn_auxiliary = cnn_backbone(input_images)

        loss: torch.Tensor; loss_score_net: torch.Tensor; loss_cnn_auxiliary: Union[torch.Tensor, None]
        loss_score_net = self.__loss_fn(model, omega, cnn_feature, batch_size)
        loss_cnn_auxiliary = None
        if cnn_auxiliary_flag:
            loss_cnn_auxiliary = F.cross_entropy(cnn_auxiliary, class_labels)
            loss = loss_score_net + loss_cnn_auxiliary
        else:
            loss = loss_score_net

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss_score_net.item(), loss_cnn_auxiliary.item() if loss_cnn_auxiliary is not None else 0.0

    def __loss_fn(self, model, omega, cnn_feature, batch_size: int, eps=1e-5):
        """The loss function for training score-based generative models.

        Parameters:
            model: A PyTorch model instance that represents a time-dependent score-based model.
            omega: A batch of training data. Shape = (batch_size, num_objs * 3)
            cnn_feature (torch.Tensor): image feature extracted by cnn, shape = (batch_size, embed_dim)
            marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
            eps: A tolerance value for numerical stability.
        """
        random_t = torch.rand(batch_size, device=self.device) * (1. - eps) + eps
        random_t = random_t.unsqueeze(-1)
        # -> (batch_size, 1)

        z = torch.randn_like(omega)
        # -> (batch_size, num_objs * 3)

        std = self.marginal_prob_std_fn(random_t)
        # -> (batch_size, 1) (shape = random_t.shape)

        perturbed_omega = copy.deepcopy(omega)  # Can't use torch.clone() because it syncs gradients
        perturbed_omega += z * std  # (batch_size, num_objs * 3) * (batch_size, 1)
        # -> (batch_size, num_objs * 3)

        score = model(perturbed_omega, cnn_feature, random_t, self.num_objs)
        # -> (batch_size, num_objs * 3)

        #loss = torch.mean(torch.sum(((score * std + z)**2).view(self.batch_size, -1), dim=-1))
        loss = (score * std + z)**2             # -> (batch_size, num_objs * 3)
        loss = torch.mean(torch.sum(loss, dim=-1))
        # -> () 0-dimension scalar
        return loss

    def test(self,
             config: dict,
             path_score_net_checkpoint: str,
             path_cnn_backbone_checkpoint: str,
             path_save_visualization: str,
             data_index: int,
             num_steps: int = 500,
             eps: float = 1e-3) -> None:
        """ Test a visualize a sample

        Args:
            config (dict): _description_
            path_state_dict (str): _description_
            path_save_visualization (str): _description_
            data_index (int): _description_
            num_steps (int, optional): the number of sampling steps.
                Defaults to 500.
            eps (float, optional): the smallest time step for numerical
                stability. Defaults to 1e-3.

        Ref: Song Yang's blog
        [link](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3)
        """
        # Expand config - dataset
        input_image_type: str = config['dataset']['input_image_type']   # 'concrete' or 'segmentation'
        # Expand config - score_net
        num_fc_layers = config['score_net']['num_fc_layers']
        hidden_dim = config['score_net']['hidden_dim']
        embed_dim = config['score_net']['embed_dim']
        # Expand config - cnn_backbone
        cnn_backbone_model: str = config['cnn_backbone']['model']

        # Loading models
        # Score-net
        print('Loading model...')
        score_net = ScoreNetTangram(marginal_prob_std_func=self.marginal_prob_std_fn,
                                    num_objs=self.num_objs,
                                    device=self.device,
                                    num_fc_layers=num_fc_layers,
                                    hidden_dim=hidden_dim,
                                    embed_dim=embed_dim)
        state_dict_score_net = torch.load(path_score_net_checkpoint)
        score_net.load_state_dict(state_dict_score_net)
        score_net.to(self.device)
        score_net.eval()
        # CNN backbone
        cnn_backbone: nn.Module = eval(cnn_backbone_model)(config=config,
                                                           embed_dim=embed_dim,
                                                           num_class=self._dataset.num_class)
        state_dict_cnn_backbone = torch.load(path_cnn_backbone_checkpoint)
        cnn_backbone.load_state_dict(state_dict_cnn_backbone)
        cnn_backbone.to(self.device)
        cnn_backbone.eval()
        print('Done.')

        omega_sequences: List[List[np.ndarray]] = []
        # Target data
        original_omega: torch.Tensor = self._dataset.data_list[data_index].omega # type: ignore
        concrete_images: np.ndarray = self._dataset.data_list[data_index].concrete_images[0].unsqueeze(0) # type: ignore
        segmentation_images: torch.Tensor = self._dataset.data_list[data_index].segmentation_images[0].unsqueeze(0) # type: ignore
        binarized_images: torch.Tensor = self._dataset.data_list[data_index].binarized_images[0].unsqueeze(0) # type: ignore
        input_images: torch.Tensor = eval(input_image_type + '_images') # 'concrete' or 'segmentation'
        # -> (7, 3)
        original_omega = original_omega.view(1, 7 * 3)
        samplers.append_omega_batch(omega_sequences, original_omega)
        # Add perturbance
        t_final: torch.Tensor = torch.tensor([0.05], device=self.device)
        z = torch.randn_like(original_omega)
        std = self.marginal_prob_std_fn(t_final)
        perturbance: torch.Tensor = z * std
        perturbed_omega: torch.Tensor = original_omega + perturbance
        samplers.append_omega_batch(omega_sequences, perturbed_omega)
        # Euler-Maruyama sampler
        result: torch.Tensor
        result = samplers.euler_maruyama_sampler(score_net_model=score_net,
                                                 cnn_backbone=cnn_backbone,
                                                 diffusion_coeff_fn=self.diffusion_coeff_fn,
                                                 input_images=input_images,
                                                 omega=perturbed_omega,
                                                 omega_sequences=omega_sequences,
                                                 t_final=t_final.item(),
                                                 num_steps=num_steps,
                                                 eps=eps)
        # Evaluation
        print('Evaluating...')
        errors: List[Dict[str, float]]
        errors = evaluator.evaluate(result, original_omega, self.device) # type: ignore
        # Save results
        print('Saving results...')
        # TODO
        print('Visualizing results...')
        for i, o in enumerate(omega_sequences):
            os.system(f'mkdir -p {os.path.join(path_save_visualization, str(i))}')
            # Quantified result
            fp = open(os.path.join(path_save_visualization, f'{i}/evaluation.txt'), 'w')
            for key in errors[i].keys():
                fp.write(f'{key}:{errors[i][key]}\n')
            fp.close()
            # Visualized result
            frames = draw_tangrams(omegas=o, canvas_length=1000)
            cv2.imwrite(os.path.join(path_save_visualization, f'{i}/gt.png'), frames[0])
            cv2.imwrite(os.path.join(path_save_visualization, f'{i}/result.png'), frames[-1])
            _concrete_img: np.ndarray = cv2.resize(concrete_images[0].cpu().numpy().astype(np.uint8), frames[0].shape[:2])
            _segmentation_img: np.ndarray = cv2.resize(segmentation_images[0].cpu().numpy().astype(np.uint8), frames[0].shape[:2])
            images_to_video(os.path.join(path_save_visualization, f'{i}/inference_process.mp4'),
                            [_concrete_img] + [_segmentation_img] + frames,
                            [30,] * 4 + [1,] * (len(frames) - 3) + [60],
                            30)
