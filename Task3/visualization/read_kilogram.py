import sys
import os
from typing import Tuple, List

import json
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import numpy as np
import cv2

sys.path.append('..')
from visualization.tangram_data_tools import rotation_matrix_2d_from_theta, transformation_matrix_from_svg_str
from visualization.tangram_data_tools import tuple_2x2_dot_3x3_homogeneous
from visualization.tangram_data_tools import vertices_to_position, vertices_to_orientation
from visualization.tangram_data_tools import PIECE_SHAPE_TEMPLATES


# TODO: There's a "transform" notation in the svg (page-A ~ page-L uses this)
def parse_tangram_svg(path_svg: str) -> Tuple[tuple, dict]:
    """
    Parameters:
        path_svg - path to the SVG
    
    Returns:
        A tuple denotes (width, height) of the canvas size.
        A dictionary, format: {"1": <list_of_points>, ..., "<polygon_id>": <list_of_points>}.

    Note:
        Assume all SVG starts from (0, 0) (at least KILOGRAM is like this)
    """
    dataset_namespace = '{http://www.w3.org/2000/svg}'
    tree = ET.parse(path_svg)
    root = tree.getroot()

    tangram = dict()
    for child in root:
        tag = child.tag.replace(dataset_namespace, '')
        if tag == 'polygon':
            polygon_id = child.attrib['id']
            points: List[tuple] = parse_points(child.attrib['points'])
            tangram[polygon_id] = points
            if 'transform' in child.attrib.keys():
                transform: np.matrix = transformation_matrix_from_svg_str(child.attrib['transform'])
                for i, p in enumerate(points):
                    points[i] = tuple_2x2_dot_3x3_homogeneous(p, transform)
    width = 0
    height = 0
    for k in tangram.keys():
        w = max([p[0] for p in tangram[k]])
        h = max([p[1] for p in tangram[k]])
        if w > width:
            width = w
        if h > height:
            height = h
    return (width, height), tangram


def parse_points(str_points: str) -> List[Tuple[float, float]]:
    points = []
    number_strings = ['', '']
    number_strings_flag = 0
    if str_points[-1] != ' ':
        str_points += ' '
    for c in str_points:
        if c == ' ' or c == ',':
            if number_strings_flag == 1:
                points += [(float(number_strings[0]), float(number_strings[1]))]
                number_strings = ['', '']
                number_strings_flag = 0
            else:
                number_strings_flag += 1
        else:
            number_strings[number_strings_flag] += c
    return points


def draw_tangram(canvas_size: List[int], 
                 standardization_coef: float,
                 tangram_shape: dict,
                 thickness: int,
                 edge_color: tuple,
                 fill_color: tuple,
                 bg_color: tuple) -> np.ndarray:
    canvas_size = [int(x) for x in canvas_size]
    canvas_size.reverse()
    img = np.ones(canvas_size + [3,]) * np.asarray(bg_color)
    for key in tangram_shape.keys():
        #os.system(f'mkdir -p separate/{key}')
        #tmp_img = np.ones(canvas_size + (3,)) * np.asarray(bg_color)

        pts = np.array(tangram_shape[key]) * standardization_coef
        pts = pts.astype(int)
        cv2.polylines(img, [pts], True, edge_color, thickness, cv2.LINE_AA)
        cv2.fillPoly(img, [pts], fill_color)

        #cv2.polylines(tmp_img, [pts], True, fill_color, thickness)
        #cv2.fillPoly(tmp_img, [pts], fill_color)
        #cv2.imwrite(f"separate/{key}/{I}.png", tmp_img)
    return img


def read_kilogram(path_dataset: str, subset: str, _template_vertices=PIECE_SHAPE_TEMPLATES) -> dict:
    r"""
    JSON schema (parts that will be used):
    tangram
    ├── snd: float (Shape Naming Divergence)
    ├── pnd: float (Part Naming Divergence)
    ├── psa: float (Part Segmentation Agreement)
    └── annotations: list
        ├── whole: dict
        │   ├── wholeAnnotation: str
        │   └── timestamp: str (actually won't be used this time)
        ├── part: dict (corresponds to SVG polygon ids)
        │   ├── '1': str
        │   ├── ...
        │   └── '7': str
        └── metadata: dict
            └── [actionIndex]: dict (in sequence of actions)
                ├── final: bool (if the part annotation is in the final submission)
                ├── pieces: list
                ├── annotation: str
                └── timestamp: str (actually won't be used this time)

    Returns:
     The structure will be a little bit different, be like:
      tangram_data: dict
      └── tangram_id: dict
          ├── snd: float (Shape Naming Divergence)
          ├── pnd: float (Part Naming Divergence)
          ├── psa: float (Part Segmentation Agreement)
          ├── tangram_shape: dict
          ├── positions: dict
          ├── orientations: dict
          ├── canvas_size: tuple
          ├── standardization_coef: float
          └── annotations: list
              ├── whole_annotation: str
              ├── part: dict (corresponds to SVG polygon ids)
              └── metadata: dict

     The `tangram_shape` is a dictionary with IDs of pieces as its keys.
     Under each ID, there's a set of points (three to four points) that
     represents polygons' vertices.
     IDs each refers to: (1) -> square; (2) -> parallelogram;
     (3) -> small triangle; (4) -> another small triangle;
     (5) -> middle-size triangle; (6) -> big triangle;
     (7) -> another big triangle
     in which (3) \cong (4) and (6) \cong (7)
     * \cong means congruent

     P.S. I changed all their camel case names to snake case names
    """
    def standardize_tangram_shape(tangram_shape: dict) -> float:
        """
        Standardize points of a tangram
        """
        def get_edge_length(square_vertices: list) -> float:
            p0 = square_vertices[0]
            p1 = square_vertices[1]
            return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
        coef: float = get_edge_length(tangram_shape['1'])
        for k in tangram_shape.keys():
            points = []
            for p in tangram_shape[k]:
                points += [(p[0] / coef, p[1] / coef)]
            tangram_shape[k] = points
        return coef

    tangram_data = dict()

    # Load metadata
    with open(os.path.join(path_dataset, f'{subset}.json')) as fp:
        json_dataset_metadata = json.load(fp)
    tangram_ids = list(json_dataset_metadata.keys())

    # Load tangram SVGs and correspond metadata
    for t_id in tangram_ids:
        tangram = dict()
        tangram['snd'] = json_dataset_metadata[t_id]['snd']
        tangram['pnd'] = json_dataset_metadata[t_id]['pnd']
        tangram['psa'] = json_dataset_metadata[t_id]['psa']
        path_svg = os.path.join(path_dataset, 'tangrams-svg', f'{t_id}.svg')
        tangram['canvas_size'], tangram['tangram_shape'] = parse_tangram_svg(path_svg)
        tangram['standardization_coef'] = standardize_tangram_shape(tangram['tangram_shape'])
        positions = dict()
        orientations = dict()
        for p_id in tangram['tangram_shape'].keys():
            positions[p_id] = vertices_to_position(tangram['tangram_shape'][p_id])
            orientations[p_id] = vertices_to_orientation(_template_vertices[p_id], tangram['tangram_shape'][p_id], positions[p_id])
        tangram['positions'] = positions
        tangram['orientations'] = orientations
        annotations = []
        for sa in json_dataset_metadata[t_id]['annotations']:
            single_annotation = dict()
            single_annotation['whole_annotation'] = sa['whole']['wholeAnnotation']
            single_annotation['part'] = sa['part']
            single_annotation['metadata'] = sa['metadata']
            annotations += [single_annotation]
        tangram['annotations'] = annotations
        tangram_data[t_id] = tangram
    return tangram_data


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 read_kilogram.py <path_to_kilogram_dataset> <subset>',
              '[path_to_save_parsed_json]')
        print('\nSubsets include:')
        print('\tfull (full.json)')
        print('\tdense (dense.json)')
        print('\tdense10 (dense10.json)')
        print('\nExample: python3 read_kilogram.py ~/kilogram/dataset/ full')
        exit(-1)

    path_kilogram_dataset = sys.argv[1]
    subset = sys.argv[2]

    tangram_data = read_kilogram(path_kilogram_dataset, subset)

    os.system('mkdir -p all_tangrams')
    for k in tangram_data.keys():
        tangram = tangram_data[k]
        # Outline & fill
        img = draw_tangram(tangram['canvas_size'],
                           tangram['standardization_coef'],
                           tangram['tangram_shape'],
                           2,
                           (0x72, 0x42, 0x57),
                           (0x98, 0x58, 0x76),
                           (0xff, 0xff, 0xff))
        # Position
        coef = tangram['standardization_coef']
        for piece_id in tangram['tangram_shape']:
            vertices = tangram['tangram_shape'][piece_id]
            p = vertices_to_position(vertices)
            p = (int(p[0] * coef), int(p[1] * coef))
            cv2.circle(img, p, 2, (0x53, 0xd0, 0x52), -1)
        # Orientation
        vec_len = 0.4 * coef
        basis_vector = np.matrix([0, 1]).T * vec_len
        for piece_id in tangram['orientations']:
            rotation = tangram['orientations'][piece_id]
            delta = rotation_matrix_2d_from_theta(rotation) * basis_vector
            p1 = (np.array(tangram['positions'][piece_id]) * coef).astype(int)
            p2 = np.array(p1 + delta.T).reshape((2,)).astype(int)
            cv2.arrowedLine(img, p1, p2, (0x53, 0xd0, 0x52),
                            thickness=2, line_type=cv2.LINE_AA,
                            shift=0, tipLength=0.2)
            pass
        cv2.imwrite(f'all_tangrams/{k}.png', img)


    ''' Save dataset into json '''
    if len(sys.argv) == 4:
        path_parsed_json = sys.argv[3]
        with open(path_parsed_json, 'w') as fp:
            parsed_json = json.dumps(tangram_data)
            fp.write(parsed_json)


    ''' Unit Tests '''
    #print('Square:')
    #for p in tangram_data['page1-26']['tangram_shape']['1']:
    #    print('(%.9f,' % p[0], end='')
    #    print(' %.9f)' % p[1])
    #print()
    #print('Parallelogram:')
    #for p in tangram_data['page1-7']['tangram_shape']['2']:
    #    print('(%.9f,' % p[0], end='')
    #    print(' %.9f)' % p[1])
    #print()
    #print('Small triangle:')
    #for p in tangram_data['page3-70']['tangram_shape']['4']:
    #    print('(%.9f,' % p[0], end='')
    #    print(' %.9f)' % p[1])
    #print()
    #print('Middle-size triangle:')
    #for p in tangram_data['page1-149']['tangram_shape']['5']:
    #    print('(%.9f,' % p[0], end='')
    #    print(' %.9f)' % p[1])
    #print()
    #print('Big triangle:')
    #for p in tangram_data['page1-1']['tangram_shape']['6']:
    #    print('(%.9f,' % p[0], end='')
    #    print(' %.9f)' % p[1])

    theta = vertices_to_orientation(PIECE_SHAPE_TEMPLATES['6'],
                                    tangram_data['page1-5']['tangram_shape']['6'],
                                    vertices_to_position(tangram_data['page1-5']['tangram_shape']['6']))
    print()
    print(theta)
