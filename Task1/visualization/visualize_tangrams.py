from typing import List, Dict, Tuple

import numpy as np
import cv2

from tangram_data_tools import rotation_matrix_2d_from_theta
from tangram_data_tools import PIECE_SHAPE_TEMPLATES

def draw_tangrams(positions: List[np.ndarray],
                  orientations: List[float],
                  canvas_length: int,
                  thickness: int,
                  edge_color: Tuple[int, int, int],
                  fill_color: Tuple[int, int, int],
                  bg_color: Tuple[int, int, int],
                  position_point_color: Tuple[int, int, int] = (0x53, 0xd0, 0x52),
                  orientation_vector_color: Tuple[int, int, int] = (0x53, 0xd0, 0x52)
                  ) -> List[np.ndarray]:
    """
    Args:
        positions (List[np.ndarray]): List of positions (normalized) (displacement
            from (0, 0), i.e. shapes shown in PIECE_SHAPE_TEMPLATES).
        orientations (List[float]): List of orientations (normalized) (delta theta
            from shapes shown in PIECE_SHAPE_TEMPLATES)
        canvas_length (int): For a set of normalized vertices,
            let L = coef * max {x} ∪ {y}. Then, L := canvas_length.
            (i.e. canvas_length sets the length of the longest side of the background)
            (NOTE: then, coef = L / max {x} ∪ {y})
        thickness (int): Thickness of edges and orientation vector. And, the radius
            of a position point of a tangram piece.
    """
    # Calculate all vertices (normalized)
    tangram_vertices: List[dict] = []
    for i, pos in enumerate(positions):
        assert(type(pos) == np.ndarray)
        assert(pos[0] < 1 and pos[1] < 1)
        assert(pos[0] > -1 and pos[1] > -1)
        theta: float = orientations[i]
        R_theta: np.matrix = rotation_matrix_2d_from_theta(theta)
        tangram: Dict[str, list] = dict()
        for piece_id in PIECE_SHAPE_TEMPLATES.keys():
            piece: List[np.ndarray] = []
            for vertex in PIECE_SHAPE_TEMPLATES[piece_id]:
                vertex = np.array(vertex)
                piece += [R_theta * (vertex + pos)]
            tangram[piece_id] = piece
        tangram_vertices += [tangram]
    # Find maximum canvas size
    max_width: float = 0.0
    max_height: float = 0.0
    for _, tangram in enumerate(tangram_vertices):
        for piece_id in tangram.keys():
            for vertex in tangram[piece_id]:
                if vertex[0] > max_width:
                    max_width = vertex[0]
                if vertex[1] > max_height:
                    max_height = vertex[1]
    # De-Normalization of vertices
    coef = canvas_length / max((max_width, max_height))
    for i, tangram in enumerate(tangram_vertices):
        for piece_id in tangram.keys():
            for j, _ in enumerate(tangram[piece_id]):
                tangram_vertices[i][piece_id][j] *= coef
    # Draw
    frames = []
    canvas_size = (int(max_height), int(max_width))
    vec_len: float = 0.1 * canvas_length
    basis_vector: np.matrix = np.matrix([0.0, 1.0]).T * vec_len
    for i, tangram in enumerate(tangram_vertices):
        img: np.ndarray = np.ones(canvas_size + (3,)) * np.asarray(bg_color)
        for piece_id in tangram.keys():
            # Outline & fill
            pts = np.asarray(tangram[piece_id]).astype(int)
            cv2.polylines(img=img,
                          pts=[pts],
                          isClosed=True,
                          color=edge_color,
                          thickness=thickness,
                          lineType=cv2.LINE_AA)
            cv2.fillPoly(img=img,
                         pts=[pts],
                         color=fill_color)
            # Position
            cv2.circle(img=img,
                       center=positions[i] * coef,
                       radius=thickness,
                       color=position_point_color,
                       thickness=-1)
            # Orientation
            delta: np.matrix = rotation_matrix_2d_from_theta(orientations[i] * np.pi) * basis_vector
            p1 = (np.array(positions[i]) * coef).astype(int)
            p2 = np.array(p1 + delta.T).reshape((2,)).astype(int)
            cv2.arrowedLine(img=img,
                            pt1=p1, pt2=p2,
                            color=orientation_vector_color,
                            thickness=thickness,
                            line_type=cv2.LINE_AA,
                            shift=0,
                            tipLength=0.2)
        frames += [img]
    return frames
