from typing import Tuple, List

import numpy as np
from itertools import permutations

TANGRAM_DATA_MAX: float = 2.0 + 7 * np.sqrt(2) + np.sqrt(5)


def rotation_matrix_2d_from_theta(theta: float) -> np.matrix:
    return np.matrix([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])


def transformation_matrix_from_svg_str(transformation_str: str) -> np.matrix:
    """
    See <https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform>

    Parameters:
        transformation_str (str): raw string from attribute "transform".
    """
    # Read string
    values: List[float] = [0.0,] * 6
    index: int = 0
    transformation_str = transformation_str[7:-1]    # 'matrix( ... )'
    number: str = ''
    for i, c in enumerate(transformation_str):
        if c == ',' or c == ' ' or i == len(transformation_str) - 1:
            if number == '':
                continue
            values[index] = float(number)
            index += 1
            number = ''
        elif c in '-.0123456789':
            number += c
        else:
            assert(c in ' ,-.0123456789')
    # Generate matrix
    return np.matrix([[values[0], values[2], values[4]],
                      [values[1], values[3], values[5]],
                      [0.0,       0.0,       1.0     ]])


def vertices_to_position(_vertices: list) -> Tuple[float, float]:
    vertices = _vertices + [_vertices[0]]
    def vertices_to_area(vertices: list) -> float:
        sum = 0.0
        for i, p in enumerate(vertices[:-1]):
            # x_i * y_{i+1}
            sum += p[0] * vertices[i+1][1]
            # - x_{i+1} * y_i
            sum -= vertices[i+1][0] * p[1]
        return sum / 2

    center_x = 0.0
    center_y = 0.0
    for i, p in enumerate(vertices[:-1]):
        # x_i + x_{i+1}
        _x = p[0] + vertices[i+1][0]
        # y_i + y_{i+1}
        _y = p[1] + vertices[i+1][1]
        # x_i * y_{i+1} - x_{i+1} * y_i
        _w = p[0] * vertices[i+1][1] - vertices[i+1][0] * p[1]
        #
        center_x += _x * _w
        center_y += _y * _w
    A = vertices_to_area(vertices)
    center_x /= 6.0 * A
    center_y /= 6.0 * A
    return (center_x, center_y)


def vertices_to_orientation(_template_vertices: list,
                            _obj_vertices: list,
                            _obj_position: Tuple[float, float]) -> float:
    """
    Parameters:
        _template_vertices: float accuracy is 1e-9

    Returns:
        Theta, size of the angle which the template rotates to get the obj's orientation.
    """
    assert(len(_template_vertices) == len(_obj_vertices))
    def get_theta_from_cos(cos: float, sin: float) -> float:
        r"""
        Return:
            Theta \in (-pi, pi]
        """
        if sin < 0:
            return -np.arccos(cos)
        else:
            return np.arccos(cos)
    def float_list_all_eq(_float_list: list) -> bool:
        if _float_list == []:
            return False
        float_list = _float_list + [_float_list[0]]
        flag = True
        for i, _ in enumerate(float_list[:-1]):
            flag = flag and np.isclose(float_list[i], float_list[i+1], rtol=1e-4)
        return flag

    result = 0
    ok_list = []
    for obj_vertices in permutations(_obj_vertices, len(_obj_vertices)):
        theta_list = []
        for tv, ov in zip(_template_vertices, obj_vertices):
            try:
                x = tv
                x_apos = ov
                d = _obj_position
                #
                det_A = x[0]**2 + x[1]**2   # Matrix A's determinant
                cos =  (x[0] * (x_apos[0] - d[0])) / det_A
                cos += (x[1] * (x_apos[1] - d[1])) / det_A
                sin =  (x[0] * (x_apos[1] - d[1])) / det_A
                sin -= (x[1] * (x_apos[0] - d[0])) / det_A
                theta = get_theta_from_cos(cos, sin)
                theta_list += [theta]
            except ValueError as e:
                break
        if float_list_all_eq(theta_list):
            result = theta_list[0]
            ok_list += [theta_list]
    #assert(len(ok_list) == 1)
    return result


def tuple_2x2_dot_3x3_homogeneous(tuple_2x2: Tuple[float, float],
                                  homogeneous: np.matrix) -> Tuple[float, float]:
    x: np.matrix = np.matrix([tuple_2x2[0], tuple_2x2[1], 1]).T
    x = homogeneous * x
    print((float(x[0]), float(x[1])))
    return (float(x[0]), float(x[1]))

# accuracy: 1e-9
# Center of shape varies
__PIECE_SHAPE_TEMPLATES__ = {
    # Square (from page1-26)
    '1': [(1.000000000, 0.000000000),
          (0.000000000, 0.000000000),
          (0.000000000, 1.000000000),
          (1.000000000, 1.000000000)],
    # Parallelogram (from page1-7)
    '2': [(0.000000000, 0.000000000),
          (1.414213562, 0.000000000),
          (2.121320344, 0.707106781),
          (0.707106781, 0.707106781)],
    # Small triangle (from page3-70)
    '3': [(0.707106781, 0.707106781),
          (1.414213562, 0.000000000),
          (0.000000000, 0.000000000)],
    # Small triangle (from page3-70)
    '4': [(0.707106781, 0.707106781),
          (1.414213562, 0.000000000),
          (0.000000000, 0.000000000)],
    # Middle-size triangle (from page1-149)
    '5': [(1.000000000, 1.000000000),
          (2.000000000, 0.000000000),
          (0.000000000, 0.000000000)],
    # Big triangle (from page1-1)
    '6': [(1.414213562, 1.414213562),
          (2.828427125, 0.000000000),
          (0.000000000, 0.000000000)],
    # Big triangle (from page1-1)
    '7': [(1.414213562, 1.414213562),
          (2.828427125, 0.000000000),
          (0.000000000, 0.000000000)],
}
PIECE_SHAPE_TEMPLATES = dict()  # Center is at (0, 0)
for i in ['1', '2', '3', '4', '5', '6', '7']:
    shape = __PIECE_SHAPE_TEMPLATES__[i]
    pos = vertices_to_position(shape)
    PIECE_SHAPE_TEMPLATES[i] = [tuple(p) for p in (np.array(shape) - np.array(pos))]