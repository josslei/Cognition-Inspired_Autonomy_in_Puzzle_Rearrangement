from typing import Tuple, List

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from itertools import permutations

TANGRAM_DATA_MAX: float = 2.0 + 7 * np.sqrt(2) + np.sqrt(5)
ABSOLUTE_TOLERANCE: float = 1e-4


def rotation_matrix_2d_from_theta(theta: float) -> np.matrix:
    if type(theta) == np.ndarray:
        theta = theta.item()
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
                            _obj_position: Tuple[float, float],
                            _id: str = 'I dont know') -> float:
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
        if cos > 1:
            if np.isclose(cos, 1, atol=ABSOLUTE_TOLERANCE):
                cos = 1
            else:
                raise ValueError(f'Value of cos is too big! cos = {cos}')
        elif cos < -1:
            if np.isclose(cos, -1, atol=ABSOLUTE_TOLERANCE):
                cos = -1
            else:
                raise ValueError(f'Value of cos is too small! cos = {cos}')
        if sin < 0:
            return -np.arccos(cos)
        else:
            return np.arccos(cos)
    def float_list_all_eq(_float_list: list) -> bool:
        if _float_list == []:
            return False
        float_list = _float_list + [_float_list[0]]
        flag: bool = True
        for i, _ in enumerate(float_list[:-1]):
            _f: bool = np.isclose(float_list[i], float_list[i+1], atol=ABSOLUTE_TOLERANCE)
            is_close_to_abs_pi: bool = np.isclose(float_list[i], np.pi, atol=ABSOLUTE_TOLERANCE)
            is_close_to_abs_pi = is_close_to_abs_pi or np.isclose(float_list[i], -np.pi, atol=ABSOLUTE_TOLERANCE)
            if is_close_to_abs_pi:
                _f = _f or np.isclose(-float_list[i], float_list[i+1], atol=ABSOLUTE_TOLERANCE)
            flag = flag and _f
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
    if len(ok_list) == 0:
        raise RuntimeError('Cannot find rotation!', _id)
    return result


def tuple_2x2_dot_3x3_homogeneous(tuple_2x2: Tuple[float, float],
                                  homogeneous: np.matrix) -> Tuple[float, float]:
    x: np.matrix = np.matrix([tuple_2x2[0], tuple_2x2[1], 1]).T
    x = homogeneous * x
    #print((float(x[0]), float(x[1])))
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
    '2': [(0.707106781, 0.000000000),
          (2.121320344, 0.000000000),
          (1.414213562, 0.707106781),
          (0.000000000, 0.707106781)],
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

'''
def procrustes(X, Y, scaling=False, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    
    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}
   
    return d, Z, tform

def vertices_to_orientation(_template_vertices: list,
                            _obj_vertices: list,
                            _obj_position: Tuple[float, float],
                            _id: str = 'I dont know') -> float:
    def difference(vertices_1: np.ndarray, vertices_2: np.ndarray) -> float:
        squared_vertices: np.ndarray = np.asarray(vertices_1 - vertices_2) ** 2
        return (np.sqrt(squared_vertices[:,0] + squared_vertices[:,1])).mean()
    def same_dir(vertices_1: np.ndarray, vertices_2: np.ndarray) -> bool:
        dir_vec_1: np.ndarray = vertices_1[1] - vertices_1[0]
        dir_vec_2: np.ndarray = vertices_2[1] - vertices_2[0]
        diff: float = np.sqrt(np.sum(np.asarray(dir_vec_1 - dir_vec_2)**2)).item()
        #if diff < 1.71875:
        if diff < 0.1:
            return True
        else:
            return False

    template_vertices: np.ndarray = np.array(_template_vertices)
    #obj_vertices: np.ndarray = np.array(_obj_vertices) - np.array(_obj_position)

    min_d: float = 1.0
    theta_best: float = float('nan')

    if _id == 'page2-34':
        print('here')
    for _ in range(2):
        for ov in permutations(_obj_vertices, len(_obj_vertices)):
            obj_vertices = np.array(ov) - np.array(_obj_position)
            #d, Z, tform = procrustes(obj_vertices, template_vertices)
            d, Z, tform = procrustes(template_vertices, obj_vertices)
            rotation_matrix = tform['rotation']
            cos = rotation_matrix[0][0]
            sin = rotation_matrix[1][0]
            if cos > 1:
                cos = 1
            elif cos < -1:
                cos = -1
            theta: float = np.arccos(cos)
            if sin < 0:
                theta = -theta

            # Verify
            vertices_verify: np.ndarray = np.dot(template_vertices, rotation_matrix_2d_from_theta(theta))
            #if difference(vertices_verify, obj_vertices) > 1:
            #    continue
            if not same_dir(vertices_verify, obj_vertices):
                continue
            if d < min_d:
                min_d = d
                theta_best = theta
        
        if theta_best >= -np.pi and theta_best <= np.pi:
            break
        # Flip the shape
        for row, v in enumerate(template_vertices):
            template_vertices[row][0] = -v[0]

    if not (theta_best >= -np.pi and theta_best <= np.pi):
        theta_best = 0
    assert theta_best >= -np.pi
    assert theta_best <= np.pi
    return theta_best

def vertices_to_orientation(_template_vertices: list,
                            _obj_vertices: list,
                            _obj_position: Tuple[float, float],
                            _id: str = 'I dont know') -> float:
    template_vertices: np.ndarray = np.array(_template_vertices)
    obj_vertices: np.ndarray = np.array(_obj_vertices) - np.array(_obj_position)

    if _id == 'page2-34':
        print('here')
    # Procrustes analysis
    mtx1, mtx2, _ = procrustes(template_vertices, obj_vertices)

    u, _, vt = np.linalg.svd(mtx2.T @ mtx1)
    rotation_matrix = vt.T @ u.T

    theta: float = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1.0, 1.0))
    assert theta >= -np.pi
    assert theta <= np.pi
    return theta

def vertices_to_orientation(_template_vertices: list,
                            _obj_vertices: list,
                            _obj_position: Tuple[float, float],
                            _id: str = 'I dont know') -> float:
    template_vertices: np.ndarray = np.array(_template_vertices)
    obj_vertices: np.ndarray = np.array(_obj_vertices) - np.array(_obj_position)

    if _id == 'page2-34':
        print('here')
    # Procrustes analysis
    rotation_matrix, _ = orthogonal_procrustes(template_vertices, obj_vertices)

    cos = rotation_matrix[0][0]
    sin = rotation_matrix[1][0]
    if cos > 1:
        cos = 1
    elif cos < -1:
        cos = -1

    theta: float = np.arccos(cos)
    if sin < 0:
        theta = -theta
    assert theta >= -np.pi
    assert theta <= np.pi
    return theta

def vertices_to_orientation(_template_vertices: list,
                            _obj_vertices: list,
                            _obj_position: Tuple[float, float],
                            _id: str = 'I dont know') -> float:
    def rotate_vertices(vertices: np.ndarray, theta: float) -> np.ndarray:
        rot_matrix: np.matrix = rotation_matrix_2d_from_theta(theta)
        return np.dot(vertices, rot_matrix)
    def objective_function(theta: float, vertices_1: np.ndarray, vertices_2: np.ndarray) -> float:
        rotated_vertices_1: np.ndarray = rotate_vertices(vertices_1, theta)
        distance = np.sum(np.square(rotated_vertices_1 - vertices_2))
        return distance

    template_vertices: np.ndarray = np.array(_template_vertices)
    obj_vertices: np.ndarray = np.array(_obj_vertices)
    # Initially estimated theta
    #initial_theta: float = 0.0
    initial_theta: np.ndarray = np.array([0])
    # Find rotation by least square method
    result = minimize(fun=objective_function,
                      x0=initial_theta,
                      args=(template_vertices, obj_vertices),
                      tol=1e-9)
                      #bounds=((-np.pi, np.pi)))
    theta = result.x[0]
    if _id == 'page2-34':
        distance = np.sum(np.square(template_vertices - obj_vertices))
        print(theta)
        print(distance)
    return theta
'''
