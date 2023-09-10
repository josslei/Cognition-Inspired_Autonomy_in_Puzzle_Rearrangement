import sys
import os
from typing import Tuple, List

import numpy as np
import cv2


def find_mean_std(path_to_image_dataset: str, subset: str) -> Tuple[np.ndarray, np.ndarray]:
    mean: np.ndarray = np.zeros((3,))
    std: np.ndarray = np.zeros((3,))
    dirs: List[str] = os.listdir(path_to_image_dataset)
    amount: int = 0
    for tangram_id in dirs:
        d: str = os.path.join(path_to_image_dataset, tangram_id)
        if not os.path.isdir(d):
            continue
        # Iterate iamges in the same tangram ID
        image_path_list: List[str] = os.listdir(d)
        for img_name in image_path_list:
            img_path: str = os.path.join(d, img_name)
            if not os.path.isfile(img_path):
                continue
            amount += 1
            # Read & convert
            img: np.ndarray = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Min-max normalization (to [0, 1])
            img = img.astype(np.float64)
            img /= 255.0
            # Calc mean & std
            for c in range(3):
                mean[c] += img[:, :, c].mean()
                std[c] += img[:, :, c].std()
            assert img is not None
            assert img.std() != 0
        pass
    mean /= amount
    std /= amount
    return mean, std


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 find_mean_std.py <path_to_image_dataset> <subset>')
        print('\nSubsets include:')
        print('\tconcrete (concrete images)')
        print('\tsegmentation (segmentation images)')
        print('\nExample: python3 find_mean_std.py ./images/ concrete')
        exit()
    
    path_to_image_dataset: str = sys.argv[1]
    subset: str = sys.argv[2]

    mean: np.ndarray; std: np.ndarray
    mean, std = find_mean_std(path_to_image_dataset=path_to_image_dataset, subset=subset)
    print(f'mean: {mean}')
    print(f'std:  {std}')
    pass
