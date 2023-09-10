from typing import Union, List, Tuple, Dict, Any

import torch
import torchvision
import numpy as np

import random
import time
from PIL import Image

from utils import masks_to_segmentation_image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Data:
    transforms: torchvision.transforms.Compose;
    @staticmethod
    def set_transforms(transforms: torchvision.transforms.Compose):
        Data.transforms = transforms

    def __init__(self,
                 id: str,
                 class_label: int,
                 omega: Union[torch.Tensor, None],
                 concrete_images: Union[List[torch.Tensor], None],
                 segmentation_masks: Union[List[List[Dict[str, Any]]], None],
                 segmentation_images: Union[List[torch.Tensor], None],
                 segmentation_random_color: bool,
                 enhanced_images: Union[List[torch.Tensor], None],
                 binarized_images: Union[List[torch.Tensor], None],
                 device: torch.device = DEVICE):
        self.id: str = id
        self.class_label: int = class_label
        self.omega: Union[torch.Tensor, None] = omega
        self.concrete_images: Union[List[torch.Tensor], None] = concrete_images
        self.segmentation_masks: Union[List[List[Dict[str, Any]]], None] = segmentation_masks
        self.segmentation_images: Union[List[torch.Tensor], None] = segmentation_images
        self.segmentation_random_color: bool = segmentation_random_color
        self.enhanced_images: Union[List[torch.Tensor], None] = enhanced_images
        self.binarized_images: Union[List[torch.Tensor], None] = binarized_images
        self.device = device

        self.__selection_count: int = 0
        self.__selection_list: List[int] = list(range(len(concrete_images))) # type: ignore
        self.__random_seed: float = time.time()

        self.__regenerate_count: int = 0

        random.seed(self.__random_seed)
        random.shuffle(self.__selection_list)

    def get_one(self) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.omega is not None
        assert self.concrete_images is not None
        assert self.segmentation_masks is not None
        assert self.segmentation_images is not None
        assert self.enhanced_images is not None
        assert self.binarized_images is not None
        assert len(self.concrete_images) == len(self.segmentation_masks) # type: ignore
        if self.__selection_count == len(self.__selection_list): # type: ignore
            random.shuffle(self.__selection_list)
            self.__selection_count = 0
            # Generate new segmentation images every 10 rounds
            self.__regenerate_count += 1
            if self.segmentation_random_color and self.__regenerate_count == 10:
                for i, m in enumerate(self.segmentation_masks):
                    _segmentation_image_np: np.ndarray; _segmentation_image: torch.Tensor
                    _segmentation_image_np = masks_to_segmentation_image(m).astype(np.uint8)
                    _segmentation_image = Data.transforms(Image.fromarray(_segmentation_image_np)).to(self.device) # type: ignore
                    self.segmentation_images[i] = _segmentation_image
                self.__regenerate_count = 0
        index: int = self.__selection_list[self.__selection_count]
        _concrete_image: torch.Tensor = self.concrete_images[index] # type: ignore
        _segmentation_masks: torch.Tensor = self.segmentation_masks[index] # type: ignore
        _segmentation_image: torch.Tensor = self.segmentation_images[index] # type: ignore
        _enhanced_image: torch.Tensor = self.enhanced_images[index] # type: ignore
        _binarized_image: torch.Tensor = self.binarized_images[index] # type: ignore
        self.__selection_count += 1
        return self.omega, _concrete_image, _segmentation_masks, _segmentation_image, _enhanced_image, _binarized_image # type: ignore
