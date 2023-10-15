import os
import numpy as np
from typing import List, Tuple, Dict, Any, Union

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

import pickle
import random
import time
import colorsys
import cv2
from PIL import Image

from data import Data
from utils import masks_to_segmentation_image, increase_contrast, foreground_extraction, binarization
from visualization.read_kilogram import read_kilogram

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (224, 224)
TRANSFORMS = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])
ITERS_PER_BATCH: int = 10


class Dataset_KILOGRAM(Dataset):
    def __init__(self, config, device=DEVICE) -> None:
        super().__init__()

        # Expand config - dataset
        path_kilogram: str = config['dataset']['path_kilogram']
        path_concrete_images: str = config['dataset']['path_concrete_images']
        input_image_type: str = config['dataset']['input_image_type']
        segmentation_random_color: bool = False
        if input_image_type == 'segmentation':
            segmentation_random_color = config['dataset']['segmentation_random_color']
        method_increase_contrast: str = config['dataset']['method_increase_contrast']
        method_binarization: str = config['dataset']['method_binarization']
        # Expand config - segmentation_model
        segmentation_model_model_type: str = config['segmentation_model']['model_type']
        segmentation_model_checkpoint: str = config['segmentation_model']['checkpoint']
        # Expand config - training
        batch_size: int = config['training']['batch_size']

        self._method_increase_contrast: str = method_increase_contrast
        self._method_binarization: str = method_binarization
        self.path_concrete_images: str = path_concrete_images
        self.device: torch.device = device
        self.num_class: int = 0
        self.batch_size: int = batch_size
        self.transforms: torchvision.transforms.Compose = TRANSFORMS
        Data.set_transforms(TRANSFORMS)

        # Segmentation model
        self.segmentation_model: nn.Module = sam_model_registry[segmentation_model_model_type](checkpoint=segmentation_model_checkpoint).to(self.device)
        self.mask_generator_medium = SamAutomaticMaskGenerator(model=self.segmentation_model,
                                                               points_per_side=16,
                                                               pred_iou_thresh=0.96,
                                                               stability_score_thresh=0.98,
                                                               min_mask_region_area=100)
        self.mask_generator_high   = SamAutomaticMaskGenerator(model=self.segmentation_model,
                                                               points_per_side=32,
                                                               pred_iou_thresh=0.96,
                                                               stability_score_thresh=0.98,
                                                               min_mask_region_area=100)
        self.mask_generator_guarantee = SamAutomaticMaskGenerator(self.segmentation_model)

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
        self.test_list: List[Data] = []
        for tk in tangram_data.keys():
            omega = []
            for pk in tangram_data[tk]['positions'].keys():
                o = list(tangram_data[tk]['positions'][pk])
                o += [tangram_data[tk]['orientations'][pk]]
                omega += [o]
            # Read & process data
            _omega: torch.Tensor = torch.tensor(omega, device=device, dtype=torch.float32)
            concrete_images: Union[List[torch.Tensor], None]
            segmentation_masks: Union[List[List[Dict[str, Any]]], None]
            segmentation_images: Union[List[torch.Tensor], None]
            enhanced_images: Union[List[torch.Tensor], None]
            binarized_images: Union[List[torch.Tensor], None]
            concrete_images, segmentation_masks, segmentation_images, enhanced_images, binarized_images = self.__prepare_images(tk)
            if type(concrete_images) is type(None) or type(segmentation_masks) is type(None):
                continue
            # Split some to test
            concrete_images_test: Union[List[torch.Tensor], None]
            segmentation_masks_test: Union[List[List[Dict[str, Any]]], None]
            segmentation_images_test: Union[List[torch.Tensor], None]
            enhanced_images_test: Union[List[torch.Tensor], None]
            binarized_images_test: Union[List[torch.Tensor], None]
            _test_len: int = int(len(concrete_images) / 5) # type: ignore
            #
            concrete_images_test = concrete_images[-_test_len:] # type: ignore
            segmentation_masks_test = segmentation_masks[-_test_len:] # type: ignore
            segmentation_images_test = segmentation_images[-_test_len:] # type: ignore
            enhanced_images_test = enhanced_images[-_test_len:] # type: ignore
            binarized_images_test = binarized_images[-_test_len:] # type: ignore
            concrete_images = concrete_images[:-_test_len] # type: ignore
            segmentation_masks = segmentation_masks[:-_test_len] # type: ignore
            segmentation_images = segmentation_images[:-_test_len] # type: ignore
            enhanced_images = enhanced_images[:-_test_len] # type: ignore
            binarized_images = binarized_images[:-_test_len] # type: ignore
            # Append to training set
            self.data_list += [Data(id=tk,
                                    class_label=self.num_class,
                                    omega=_omega,
                                    concrete_images=concrete_images,
                                    segmentation_masks=segmentation_masks,
                                    segmentation_images=segmentation_images,
                                    segmentation_random_color=segmentation_random_color,
                                    enhanced_images=enhanced_images,
                                    binarized_images=binarized_images,
                                    device=self.device)]
            # Append to testing set
            self.test_list += [Data(id=tk,
                                    class_label=self.num_class,
                                    omega=_omega,
                                    concrete_images=concrete_images_test,
                                    segmentation_masks=segmentation_masks_test,
                                    segmentation_images=segmentation_images_test,
                                    segmentation_random_color=segmentation_random_color,
                                    enhanced_images=enhanced_images_test,
                                    binarized_images=binarized_images_test,
                                    device=self.device)]
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

        # Visualize to verify
        '''
        from visualization.visualize_tangrams import draw_tangrams
        om = [o.omega.cpu().numpy() for o in self.data_list]
        frames = draw_tangrams(omegas=om, canvas_length=1000)
        os.system('mkdir -p tmp_images')
        for i, f in enumerate(frames):
            cv2.imwrite(f'tmp_images/{i}.png', f)
        print('done.')
        exit()
        '''

        # Trick the dataloader to make the dataset recursively iterable
        self.data_list = self.data_list * int(batch_size / len(self.data_list))
        self.data_list = self.data_list * ITERS_PER_BATCH   # Make ITERS_PER_BATCH iters an epoch
        #
        del self.segmentation_model
        del self.mask_generator_medium
        del self.mask_generator_high
        del self.mask_generator_guarantee
        torch.cuda.empty_cache()
        return

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Omega
                Shape of omega: (7, 3)
        """
        assert self.data_list[index].omega is not None
        assert self.data_list[index].segmentation_masks is not None
        _omega: torch.Tensor
        _concrete_image: torch.Tensor
        _segmentation_masks: List[Dict[str, Any]]
        _segmentation_image: torch.Tensor
        _enhanced_image: torch.Tensor
        _binarized_image: torch.Tensor
        _class_label: int
        _omega, _concrete_image, _segmentation_masks, _segmentation_image, _enhanced_image, _binarized_image = self.data_list[index].get_one()
        _class_label = self.data_list[index].class_label
        return _omega, _concrete_image, _segmentation_image, _enhanced_image, _binarized_image, _class_label

    def __len__(self) -> int:
        return len(self.data_list)

    def __prepare_images(self, data_id: str) -> Tuple[Union[List[torch.Tensor], None],
                                                      Union[List[List[Dict[str, Any]]], None],
                                                      Union[List[torch.Tensor], None],
                                                      Union[List[torch.Tensor], None],
                                                      Union[List[torch.Tensor], None]]:
        print(f'preparing for id: {data_id}')
        concrete_images_root: str = os.path.join(self.path_concrete_images, data_id)
        segmentation_images_root: str = concrete_images_root + '.segmentation'
        enhanced_images_root: str = concrete_images_root + '.enhanced'
        binarized_images_root: str = concrete_images_root + '.binarized'
        if not os.path.isdir(concrete_images_root):
            return None, None, None, None, None
        else:
            os.system(f'mkdir -p {segmentation_images_root}')
            os.system(f'mkdir -p {enhanced_images_root}')
            os.system(f'mkdir -p {binarized_images_root}')
        image_names_list: List[str] = [_f for _f in os.listdir(concrete_images_root)]
        concrete_images_list: List[torch.Tensor] = []
        segmentation_images_list: List[torch.Tensor] = []
        segmentation_masks_list: List[List[Dict[str, Any]]] = []
        enhanced_images_list: List[torch.Tensor] = []
        binarized_images_list: List[torch.Tensor] = []
        for _img_name in image_names_list:
            _concrete_image_path: str = os.path.join(concrete_images_root, _img_name)
            _segmentation_masks_path: str = os.path.join(segmentation_images_root, _img_name[:_img_name.find('.')] + '.pkl')
            _segmentation_image_path: str = os.path.join(segmentation_images_root, _img_name)
            _enhanced_image_path: str = os.path.join(enhanced_images_root, _img_name)
            _binarized_image_path: str = os.path.join(binarized_images_root, _img_name)
            # Read the concrete image
            _concrete_image: np.ndarray = cv2.imread(_concrete_image_path)
            _concrete_image = cv2.cvtColor(_concrete_image, cv2.COLOR_BGR2RGB)
            _preprocessed_concrete_image: torch.Tensor = self.transforms(Image.fromarray(_concrete_image)) # type: ignore
            concrete_images_list += [_preprocessed_concrete_image.to(self.device)]
            # Read/generate the segmentation image
            masks: List[Dict[str, Any]]
            if not os.path.isfile(_segmentation_masks_path):
                # Generate
                masks = self.mask_generator_medium.generate(_concrete_image)
                if len(masks) < 3:
                    masks = self.mask_generator_high.generate(_concrete_image)
                if len(masks) < 3:
                    masks = self.mask_generator_guarantee.generate(_concrete_image)
                # Cache the segmentation masks - segmenting is too slow!
                with open(_segmentation_masks_path, 'wb') as fp:
                    pickle.dump(masks, fp)
                # Generate a sample image
                _segmentation_sample_image = masks_to_segmentation_image(masks).astype(np.uint8)
                _segmentation_sample_image = cv2.cvtColor(_segmentation_sample_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(_segmentation_image_path, _segmentation_sample_image)
            else:
                # Read masks
                fp = open(_segmentation_masks_path, 'rb')
                masks = pickle.load(fp)
                fp.close()
                # Read image
                _segmentation_image = cv2.imread(_segmentation_image_path)
                _segmentation_image = cv2.cvtColor(_segmentation_image, cv2.COLOR_BGR2RGB)
            _preprocessed_segmentation_image: torch.Tensor = self.transforms(Image.fromarray(_segmentation_image)) # type: ignore
            segmentation_images_list += [_preprocessed_segmentation_image.to(self.device)]
            segmentation_masks_list += [masks]
            # Read/generate enhanced image
            _enhanced_image: np.ndarray
            if not os.path.isfile(_enhanced_image_path):
                # Generate
                _enhanced_image = increase_contrast(_concrete_image, self._method_increase_contrast)
                _enhanced_image = foreground_extraction(_concrete_image)
                cv2.imwrite(_enhanced_image_path, cv2.cvtColor(_enhanced_image, cv2.COLOR_RGB2BGR))
            else:
                # Read
                _enhanced_image = cv2.imread(_enhanced_image_path)
                _enhanced_image = cv2.cvtColor(_enhanced_image, cv2.COLOR_BGR2RGB)
            _preprocessed_enhanced_image: torch.Tensor = self.transforms(Image.fromarray(_enhanced_image)) # type: ignore
            enhanced_images_list += [_preprocessed_enhanced_image.to(self.device)]
            # Read/generate binarized image
            _binarized_image: np.ndarray
            if not os.path.isfile(_binarized_image_path):
                # Generate
                _binarized_image = binarization(_concrete_image, self._method_binarization)
                _binarized_image = _binarized_image.reshape(_binarized_image.shape + (1,))
                _binarized_image = _binarized_image.repeat(3, axis=2)
                cv2.imwrite(_binarized_image_path, cv2.cvtColor(_binarized_image, cv2.COLOR_RGB2BGR))
                pass
            else:
                # Read
                _binarized_image = cv2.imread(_binarized_image_path)
                _binarized_image = cv2.cvtColor(_binarized_image, cv2.COLOR_BGR2RGB)
            _preprocessed_binarized_image: torch.Tensor = self.transforms(Image.fromarray(_binarized_image)) # type: ignore
            binarized_images_list += [_preprocessed_binarized_image.to(self.device)]
        return concrete_images_list, segmentation_masks_list, segmentation_images_list, enhanced_images_list, binarized_images_list


class Dataset_KILOGRAM_test(Dataset):
    def __init__(self, dataset_kilogram: Dataset_KILOGRAM) -> None:
        super().__init__()

        self.test_list: List[Data] = dataset_kilogram.test_list

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Omega
                Shape of omega: (7, 3)
        """
        assert self.test_list[index].omega is not None
        assert self.test_list[index].segmentation_masks is not None
        _omega: torch.Tensor
        _concrete_image: torch.Tensor
        _segmentation_masks: List[Dict[str, Any]]
        _segmentation_image: torch.Tensor
        _enhanced_image: torch.Tensor
        _binarized_image: torch.Tensor
        _class_label: int
        _omega, _concrete_image, _segmentation_masks, _segmentation_image, _enhanced_image, _binarized_image = self.test_list[index].get_one()
        _class_label = self.test_list[index].class_label
        return _omega, _concrete_image, _segmentation_image, _enhanced_image, _binarized_image, _class_label

    def __len__(self) -> int:
        return len(self.test_list)
