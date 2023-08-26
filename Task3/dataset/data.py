from typing import Union, List, Tuple

import torch
import numpy as np

import random
import time


class Data:
    def __init__(self,
                 id: str,
                 class_label: int,
                 omega: Union[torch.Tensor, None],
                 concrete_images: Union[List[torch.Tensor], None],
                 segmentation_images: Union[List[torch.Tensor], None]):
        self.id: str = id
        self.class_label: int = class_label
        self.omega: Union[torch.Tensor, None] = omega
        self.concrete_images: Union[List[torch.Tensor], None] = concrete_images
        self.segmentation_images: Union[List[torch.Tensor], None] = segmentation_images

        self.__selection_count: int = 0
        self.__selection_list: List[int] = list(range(len(concrete_images))) # type: ignore
        self.__random_seed: float = time.time()

        random.seed(self.__random_seed)
        random.shuffle(self.__selection_list)

    def get_one(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.omega is not None
        assert self.concrete_images is not None
        assert self.segmentation_images is not None
        assert len(self.concrete_images) == len(self.segmentation_images) # type: ignore
        if self.__selection_count == len(self.__selection_list): # type: ignore
            random.shuffle(self.__selection_list)
            self.__selection_count = 0
        _concrete_image: torch.Tensor = self.concrete_images[self.__selection_list[self.__selection_count]] # type: ignore
        _segmentation_image: torch.Tensor = self.segmentation_images[self.__selection_list[self.__selection_count]] # type: ignore
        self.__selection_count += 1
        return self.omega, _concrete_image, _segmentation_image # type: ignore