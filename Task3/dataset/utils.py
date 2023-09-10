from typing import List, Dict, Any

import numpy as np
import cv2

import time
import random
import colorsys


def masks_to_segmentation_image(masks: List[Dict[str, Any]]) -> np.ndarray:
    ''' masks[i].keys():
            ['segmentation', 'area', 'bbox', 'predicted_iou',
                'point_coords', 'stability_score', 'crop_box']
    '''
    _sorted_masks: List[Dict[str, Any]] = sorted(masks, key=(lambda x: x['area']), reverse=True)
    _img: np.ndarray = np.zeros((_sorted_masks[0]['segmentation'].shape[0], _sorted_masks[0]['segmentation'].shape[1], 3))
    _colors: list = __ncolors(len(masks))
    for i, mask in enumerate(_sorted_masks):
        _m: np.ndarray = mask['segmentation']
        _mask_color: np.ndarray = np.asarray(_colors[i])
        _img[_m] = _mask_color
    return _img

def __ncolors(num) -> list:
    """
    Ref: https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
    Ref: https://github.com/choumin/ncolors/blob/master/ncolors.py
    """
    rgb_colors = []
    if num < 1:
        return rgb_colors 
    hls_colors = __get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors

def __get_n_hls_colors(num) -> list:
    """
    Ref: https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
    Ref: https://github.com/choumin/ncolors/blob/master/ncolors.py
    """
    random.seed(time.time())
    hls_colors: list = []
    i = 0
    step = 360.0 / num 
    offset = random.random() * 360
    while i < 360 + offset:
        h = i + offset
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors

def increase_contrast(input_image: np.ndarray, method: str = 'normalize') -> np.ndarray:
    """

    Args:
        input_image (np.ndarray): Expected channel sequence RGB.
        method (str, optional): _description_. Defaults to 'normalize'.

    Raises:
        ValueError: will be raised when unexpected method string is passed in.

    Returns:
        np.ndarray: Channel sequence will be as like the input.
    """
    output_image: np.ndarray = np.zeros_like(input_image)
    if method == 'normalize':
        cv2.normalize(src=input_image,
                      dst=output_image,
                      alpha=0,
                      beta=255,
                      norm_type=cv2.NORM_MINMAX,
                      dtype=cv2.CV_8U) # type: ignore
    elif method == 'equalize_hist':
        r: np.ndarray; g: np.ndarray; b: np.ndarray
        r, g, b = cv2.split(input_image)
        r1 = cv2.equalizeHist(r)
        g1 = cv2.equalizeHist(g)
        b1 = cv2.equalizeHist(b)
        output_image = cv2.merge([r1, g1, b1])
    else:
        raise ValueError('Unexpected method to increase contrast')
    return output_image
 
def foreground_extraction(input_image: np.ndarray) -> np.ndarray:
    output_image: np.ndarray

    rect: tuple = (1, 1, input_image.shape[1], input_image.shape[0])
    mask: np.ndarray = np.zeros(input_image.shape[:2], dtype=np.uint8)

    tmp_array_bgd_model = np.zeros((1, 65), np.float64)
    tmp_array_fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img=input_image,
                mask=mask,
                rect=rect,
                bgdModel=tmp_array_bgd_model,
                fgdModel=tmp_array_fgd_model,
                iterCount=5,
                mode=cv2.GC_INIT_WITH_RECT)

    mask_a = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8').reshape(mask.shape + (1,))
    mask_b = np.where((mask == 2) | (mask == 0), 255, 0).astype('uint8').reshape(mask.shape + (1,))
    output_image = input_image * mask_a + mask_b

    return output_image

def binarization(input_image: np.ndarray, method: str = 'otsu'):
    output_image: np.ndarray
    if method == 'otsu':
        input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        _, output_image = cv2.threshold(input_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError('Unexpected method to increase contrast')
    return output_image