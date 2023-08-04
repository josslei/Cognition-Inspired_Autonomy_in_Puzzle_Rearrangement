import os
import functools
import numpy as np

from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from typing import List, Tuple, Dict, Any
from PIL import Image
import copy
import pickle
import cv2

import samplers
from network.score_nets import ScoreNetTangram
from network.sde import marginal_prob_std, diffusion_coeff
from visualization.read_kilogram import read_kilogram
from visualization.visualize_tangrams import draw_tangrams, images_to_video

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from network.vgg16 import VGG16

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TarGF_Tangram:
    """ Wraps the DSM (Denoising Score-Matching) model's training & inference
    """

    def __init__(self,
                 config: dict,
                 sigma: float = 25,
                 num_objs: int = 7,
                 is_json: bool = False,
                 betas: Tuple[float, float] = (0.5, 0.999),
                 device: torch.device = DEVICE) -> None:
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device=device)
        self.num_objs = num_objs

        # Parameter for training
        self.betas = betas

        self.device = device

        # Prepare dataset
        _transforms = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            #transforms.RandomRotation(360, fill=0xff)
        ])
        # All data are loaded into RAM (of self.device) here
        print('Loading dataset...')
        self._dataset = Dataset_KILOGRAM(config, _transforms, self.device)
        print('Done.')

    def train(self, config: dict, log_save_dir: str) -> None:
        # Expand config - score_net
        num_fc_layers: int = config['score_net']['num_fc_layers']
        hidden_dim: int = config['score_net']['hidden_dim']
        embed_dim: int = config['score_net']['embed_dim']
        # Expand config - cnn_backbone
        cnn_backbone_model: str = config['cnn_backbone']['model']
        # Expand config - training
        learning_rate: float = config['training']['learning_rate']
        num_epochs: int = config['training']['num_epochs']
        batch_size: int = config['training']['batch_size']

        # Create dataloader
        self.dataloader = DataLoader(self._dataset, batch_size=batch_size, shuffle=True)

        # Create models and optimizer
        print('Creating models and optimizer...')
        score_net = ScoreNetTangram(marginal_prob_std_func=self.marginal_prob_std_fn,
                                    num_objs=self.num_objs,
                                    device=self.device,
                                    num_fc_layers=num_fc_layers,
                                    hidden_dim=hidden_dim,
                                    embed_dim=embed_dim)
        score_net.to(self.device)
        cnn_backbone: nn.Module = eval(cnn_backbone_model)(embed_dim=embed_dim)
        optimizer = optim.Adam(score_net.parameters(), lr=learning_rate, betas=self.betas)
        print('Done.')

        # Training loop
        losses_per_epoch = []
        model_dict_path: str = ''
        for epoch in trange(num_epochs):
            # Iterate batches
            for _, data in enumerate(self.dataloader):
                omega: torch.Tensor = data[0].view(batch_size, self.num_objs * 3)
                segmentation_images: torch.Tensor = data[1]
                loss: float = self.__train_one_epoch(model=score_net,
                                                     cnn_backbone=cnn_backbone,
                                                     omega=omega,
                                                     segmentation_images=segmentation_images,
                                                     optimizer=optimizer,
                                                     batch_size=batch_size)
                losses_per_epoch += [loss]
            # TODO: Evaluation
            # Save model
            if (epoch + 1) % 500 == 0:
                os.system(f'mkdir -p {log_save_dir}')
                if os.path.exists(model_dict_path):
                    os.remove(model_dict_path)
                model_dict_path = os.path.join(log_save_dir, f'score_net_epoch_{epoch}.pt')
                torch.save(score_net.state_dict(), model_dict_path)
                with open(os.path.join(log_save_dir, 'training_log_losses.txt'), 'w') as fp:
                    for loss in losses_per_epoch:
                        fp.write(f'{loss}\n')
            pass
        pass

    def __train_one_epoch(self, model, cnn_backbone, omega, segmentation_images, optimizer, batch_size: int) -> float:
        cnn_feature: torch.Tensor = cnn_backbone(segmentation_images)

        loss = self.__loss_fn(model, omega, cnn_feature, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def __loss_fn(self, model, omega, cnn_feature, batch_size: int, eps=1e-5):
        """The loss function for training score-based generative models.

        Parameters:
            model: A PyTorch model instance that represents a time-dependent score-based model.
            omega: A batch of training data. Shape = (batch_size, num_objs * 3)
            cnn_feature (torch.Tensor): image feature extracted by cnn, shape = (batch_size, embed_dim)
            marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
            eps: A tolerance value for numerical stability.
        """
        omega = omega.to(self.device)
        # -> (batch_size, num_objs * 3)

        random_t = torch.rand(batch_size, device=self.device) * (0.05) + eps
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
             path_state_dict: str,
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
        # Expand config - score_net
        num_fc_layers = config['score_net']['num_fc_layers']
        hidden_dim = config['score_net']['hidden_dim']
        embed_dim = config['score_net']['embed_dim']

        # Loading model
        print('Loading model...')
        score_net = ScoreNetTangram(marginal_prob_std_func=self.marginal_prob_std_fn,
                                    num_objs=self.num_objs,
                                    device=self.device,
                                    num_fc_layers=num_fc_layers,
                                    hidden_dim=hidden_dim,
                                    embed_dim=embed_dim)
        score_net.to(self.device)
        state_dict = torch.load(path_state_dict)
        score_net.load_state_dict(state_dict)
        score_net.eval()
        print('Done.')

        omega_sequences: List[List[np.ndarray]] = []
        # Target data
        original_omega: torch.Tensor = self._dataset.dataset_omega[data_index]  # (7, 3)
        original_omega = original_omega.view(1, 7 * 3)
        samplers.append_omega_batch(omega_sequences, original_omega)
        # Add perturbance
        t_final = 0.05
        z = torch.randn_like(original_omega)
        std = self.marginal_prob_std_fn(t_final)
        perturbance: torch.Tensor = z * std
        perturbed_omega: torch.Tensor = original_omega + perturbance
        samplers.append_omega_batch(omega_sequences, perturbed_omega)
        # Euler-Maruyama sampler
        samplers.euler_maruyama_sampler(model=score_net,
                                        diffusion_coeff_fn=self.diffusion_coeff_fn,
                                        omega=perturbed_omega,
                                        omega_sequences=omega_sequences,
                                        t_final=t_final,
                                        num_steps=num_steps,
                                        eps=eps)
        # Save results
        print('Saving results...')
        # TODO
        print('Visualizing results...')
        for i, o in enumerate(omega_sequences):
            frames = draw_tangrams(omegas=o, canvas_length=1000)
            cv2.imwrite(os.path.join(path_save_visualization, f'{i}/result.png'), frames[-1])
            images_to_video(os.path.join(path_save_visualization, f'{i}/inference_process.mp4'),
                            frames,
                            [30, 30] + [1,] * (len(frames) - 3) + [60],
                            30)


class Dataset_KILOGRAM(Dataset):
    def __init__(self, config, transforms, device=DEVICE) -> None:
        super().__init__()

        # Expand config - dataset
        path_kilogram: str = config['dataset']['path_kilogram']
        path_concrete_images: str = config['dataset']['path_concrete_images']
        # Expand config - segmentation_model
        segmentation_model_model_type: str = config['segmentation_model']['model_type']
        segmentation_model_checkpoint: str = config['segmentation_model']['checkpoint']

        self.transforms = transforms
        self.path_concrete_images: str = path_concrete_images
        self.device = device

        # Segmentation model
        self.segmentation_model: nn.Module = sam_model_registry[segmentation_model_model_type](checkpoint=segmentation_model_checkpoint)
        self.mask_generator = SamAutomaticMaskGenerator(self.segmentation_model)

        # Read tangram data
        tangram_data: dict = dict()
        if os.path.isfile(os.path.join(path_kilogram, 'tangram_data.pkl')):
            # Read parsed data
            with open(os.path.join(path_kilogram, 'tangram_data.pkl'), 'rb') as fp:
                tangram_data = pickle.load(fp)
        else:
            tangram_data = read_kilogram(path_dataset=path_kilogram, subset='full')
            # Cache parsed data
            with open(os.path.join(path_kilogram, 'tangram_data.pkl'), 'wb') as fp:
                pickle.dump(tangram_data, fp)

        # Data preprocess
        self.dataset_omega: List[torch.Tensor] = []
        self.dataset_ids: List[str] = []
        for tk in tangram_data.keys():
            omega = []
            for pk in tangram_data[tk]['positions'].keys():
                o = list(tangram_data[tk]['positions'][pk])
                o += [tangram_data[tk]['orientations'][pk]]
                omega += [o]
            _omega: torch.Tensor = torch.tensor(omega, device=device)
            self.dataset_omega += [_omega]
            self.dataset_ids += [tk]
        # Normalization
        self.DATA_MAX = 2 + 7 * np.sqrt(2) + np.sqrt(5)
        for i, _ in enumerate(self.dataset_omega):
            # Normalize positions and orientations
            for piece_id, piece in enumerate(self.dataset_omega[i]):
                self.dataset_omega[i][piece_id][0] = piece[0] / self.DATA_MAX
                self.dataset_omega[i][piece_id][1] = piece[1] / self.DATA_MAX
                self.dataset_omega[i][piece_id][2] = piece[2] / np.pi
        pass

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Omega
                Shape of omega: (7, 3)
        """
        segmentation_path: str = os.path.join(self.path_concrete_images, self.dataset_ids[index], '0_segmentation.png')
        segmentation_image: np.ndarray = np.array([])
        # Generate segmentation if not exist
        if not os.path.isfile(segmentation_path):
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            concrete_img_path: str = os.path.join(self.path_concrete_images, self.dataset_ids[index], '0.png')
            concrete_img: np.ndarray = cv2.imread(concrete_img_path)
            concrete_img = cv2.cvtColor(concrete_img, cv2.COLOR_BGR2RGB)
            masks: List[Dict[str, Any]] = self.mask_generator.generate(concrete_img)
            # Create image with segmentation info only
            segmentation_image = self.__masks_to_segmentation_image(masks)
            # Cache the segmentation image - segmenting is too slow!
            cv2.imwrite(segmentation_path, segmentation_image)
        else:
            segmentation_image = cv2.imread(segmentation_path)
        return self.dataset_omega[index], self.transforms(Image.fromarray(segmentation_image))

    def __len__(self) -> int:
        return len(self.dataset_omega)
    
    def __masks_to_segmentation_image(self, masks: List[Dict[str, Any]]) -> np.ndarray:
        pass
