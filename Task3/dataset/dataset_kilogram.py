import os
import numpy as np
from typing import List, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pickle
import random
import colorsys
import cv2
from PIL import Image

from data import Data
from visualization.read_kilogram import read_kilogram

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset_KILOGRAM(Dataset):
    def __init__(self, config, transforms, device=DEVICE) -> None:
        super().__init__()

        # Expand config - dataset
        path_kilogram: str = config['dataset']['path_kilogram']
        path_concrete_images: str = config['dataset']['path_concrete_images']
        # Expand config - segmentation_model
        segmentation_model_model_type: str = config['segmentation_model']['model_type']
        segmentation_model_checkpoint: str = config['segmentation_model']['checkpoint']
        # Expand config - training
        batch_size: int = config['training']['batch_size']

        self.transforms = transforms
        self.path_concrete_images: str = path_concrete_images
        self.device = device
        self.num_class: int = 0

        # Segmentation model
        self.segmentation_model: nn.Module = sam_model_registry[segmentation_model_model_type](checkpoint=segmentation_model_checkpoint).to(self.device)
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
        self.data_list: List[Data] = []
        for tk in tangram_data.keys():
            omega = []
            for pk in tangram_data[tk]['positions'].keys():
                o = list(tangram_data[tk]['positions'][pk])
                o += [tangram_data[tk]['orientations'][pk]]
                omega += [o]
            _omega: torch.Tensor = torch.tensor(omega, device=device, dtype=torch.float32)
            concrete_images: Union[List[torch.Tensor], None]; segmentation_images: Union[List[torch.Tensor], None]
            concrete_images, segmentation_images = self.__prepare_images(tk)
            if type(concrete_images) is type(None) or type(segmentation_images) is type(None):
                continue
            self.data_list += [Data(id=tk,
                                    class_label=self.num_class,
                                    omega=_omega,
                                    concrete_images=concrete_images,
                                    segmentation_images=segmentation_images)]
            self.num_class += 1
        # Normalization
        self.DATA_MAX = 2 + 7 * np.sqrt(2) + np.sqrt(5)
        for i, _ in enumerate(self.data_list):
            # Normalize positions and orientations
            assert self.data_list[i].omega is not None
            for piece_id, piece in enumerate(self.data_list[i].omega): # type: ignore
                self.data_list[i].omega[piece_id][0] = piece[0] / self.DATA_MAX # type: ignore
                self.data_list[i].omega[piece_id][1] = piece[1] / self.DATA_MAX # type: ignore
                self.data_list[i].omega[piece_id][2] = piece[2] / np.pi         # type: ignore

        # Trick the dataloader to make the dataset recursively iterable
        self.data_list = self.data_list * int(batch_size / len(self.data_list))
        self.data_list = self.data_list * 10    # Make 10 iters an epoch
        #
        del self.segmentation_model
        del self.mask_generator
        torch.cuda.empty_cache()
        return

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Omega
                Shape of omega: (7, 3)
        """
        assert self.data_list[index].omega is not None
        assert self.data_list[index].segmentation_images is not None
        _omega: torch.Tensor; _concrete_image: torch.Tensor; _segmentation_image: torch.Tensor; _class_label: int
        _omega, _concrete_image, _segmentation_image = self.data_list[index].get_one()
        _class_label = self.data_list[index].class_label
        return _omega, _concrete_image, _segmentation_image, _class_label

    def __len__(self) -> int:
        return len(self.data_list)

    def __prepare_images(self, data_id: str) -> Tuple[Union[List[torch.Tensor], None], Union[List[torch.Tensor], None]]:
        concrete_images_root: str = os.path.join(self.path_concrete_images, data_id)
        segmentation_images_root: str = concrete_images_root + '.segmentation'
        if not os.path.isdir(concrete_images_root):
            return None, None
        else:
            os.system(f'mkdir -p {segmentation_images_root}')
        image_names_list: List[str] = [_f for _f in os.listdir(concrete_images_root)]
        concrete_images_list: List[torch.Tensor] = []
        segmentation_images_list: List[torch.Tensor] = []
        for _img_name in image_names_list:
            _concrete_image_path: str = os.path.join(concrete_images_root, _img_name)
            _segmentation_image_path: str = os.path.join(segmentation_images_root, _img_name)
            # Read the concrete image
            _concrete_image: np.ndarray = cv2.imread(_concrete_image_path)
            _concrete_image = cv2.cvtColor(_concrete_image, cv2.COLOR_BGR2RGB)
            concrete_images_list += [self.transforms(Image.fromarray(_concrete_image)).to(self.device)]
            # Read/generate the segmentation image
            _segmentation_image: np.ndarray
            if not os.path.isfile(_segmentation_image_path):
                # Generate
                masks: List[Dict[str, Any]] = self.mask_generator.generate(_concrete_image)
                # Create image with segmentation info only
                _segmentation_image = self.__masks_to_segmentation_image(masks).astype(np.uint8)
                # Cache the segmentation image - segmenting is too slow!
                cv2.imwrite(_segmentation_image_path, _segmentation_image)
            else:
                # Read
                _segmentation_image = cv2.imread(_segmentation_image_path)
                _segmentation_image = cv2.cvtColor(_segmentation_image, cv2.COLOR_BGR2RGB)
            segmentation_images_list += [self.transforms(Image.fromarray(_segmentation_image)).to(self.device)]
        return concrete_images_list, segmentation_images_list

    def __masks_to_segmentation_image(self, masks: List[Dict[str, Any]]) -> np.ndarray:
        ''' masks[i].keys():
                ['segmentation', 'area', 'bbox', 'predicted_iou',
                 'point_coords', 'stability_score', 'crop_box']
        '''
        _sorted_masks: List[Dict[str, Any]] = sorted(masks, key=(lambda x: x['area']), reverse=True)
        _img: np.ndarray = np.zeros((_sorted_masks[0]['segmentation'].shape[0], _sorted_masks[0]['segmentation'].shape[1], 3))
        _colors: list = self.__ncolors(len(masks))
        for i, mask in enumerate(_sorted_masks):
            _m: np.ndarray = mask['segmentation']
            _mask_color: np.ndarray = np.asarray(_colors[i])
            _img[_m] = _mask_color
        return _img

    def __ncolors(self, num) -> list:
        """
        Ref: https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
        Ref: https://github.com/choumin/ncolors/blob/master/ncolors.py
        """
        rgb_colors = []
        if num < 1:
            return rgb_colors 
        hls_colors = self.__get_n_hls_colors(num)
        for hlsc in hls_colors:
            _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
            r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
            rgb_colors.append([r, g, b])

        return rgb_colors

    def __get_n_hls_colors(self, num) -> list:
        """
        Ref: https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
        Ref: https://github.com/choumin/ncolors/blob/master/ncolors.py
        """
        random.seed(1)
        hls_colors: list = []
        i = 0
        step = 360.0 / num 
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            _hlsc = [h / 360.0, l / 100.0, s / 100.0]
            hls_colors.append(_hlsc)
            i += step

        return hls_colors