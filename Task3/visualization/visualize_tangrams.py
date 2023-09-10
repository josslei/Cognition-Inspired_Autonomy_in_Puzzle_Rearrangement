from typing import List, Dict, Tuple

import numpy as np
import cv2

from visualization.tangram_data_tools import rotation_matrix_2d_from_theta
from visualization.tangram_data_tools import PIECE_SHAPE_TEMPLATES, TANGRAM_DATA_MAX


def images_to_video(path_to_save_video: str,
                    frames: List[np.ndarray],
                    repeat_times: List[int],
                    fps: int) -> None:
    """
    Parameters:
        path_to_save_video (str): path to save the video
        frames (List[np.ndarray]): frames (images)
        repeat_times (List[int]): how many times does each image in 'frames'
            repeats. For example, if repeat_times[0] is 30, frames[0]
            will be repeated for 30 times in the video.
        fps (int): frames per second
    """
    if path_to_save_video[-4:] != '.mp4':
        path_to_save_video += '.mp4'
    frame_size: Tuple[int, int] = (frames[0].shape[1], frames[0].shape[0])
    video: cv2.VideoWriter = cv2.VideoWriter(path_to_save_video,
                                             cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps,
                                             frame_size)
    for i, img in enumerate(frames):
        for _ in range(repeat_times[i]):
            video.write(img)
    video.release()


def draw_tangrams(omegas: List[np.ndarray],
                  canvas_length: int,
                  thickness: int = 2,
                  edge_color: Tuple[int, int, int] = (0x72, 0x42, 0x57),
                  fill_color: Tuple[int, int, int] = (0x98, 0x58, 0x76),
                  background_color: Tuple[int, int, int] = (0xff, 0xff, 0xff),
                  position_point_color: Tuple[int, int, int] = (0x53, 0xd0, 0x52),
                  orientation_vector_color: Tuple[int, int, int] = (0x53, 0xd0, 0x52)
                  ) -> List[np.ndarray]:
    # TODO: Update
    """
    Args:
        omegas (List[np.ndarray]): List of tangram states (x, y, theta).
            Shape = (number of frames, 7, 3)
        canvas_length (int): For a set of normalized vertices,
            let L = coef * max {x} ∪ {y}. Then, L := canvas_length.
            (i.e. canvas_length sets the length of the longest side of the background)
            (NOTE: then, coef = L / max {x} ∪ {y})
        thickness (int): Thickness of edges and orientation vector. And, the radius
            of a position point of a tangram piece.
        edge_color (Tuple[int, int, int]): Color of shapes' edges
        fill_color (Tuple[int, int, int]): Color to fill each piece of tangram
        background_color (Tuple[int, int, int]): Color of the canvas background
        position_point_color (Tuple[int, int, int], optional): The color to indicate
            positions of tangram pieces. Defaults to (0x53, 0xd0, 0x52).
        orientation_vector_color (Tuple[int, int, int], optional): The color to
            indicate orientation vector of tangram pieces.
            Defaults to (0x53, 0xd0, 0x52).
        orientation_vector_length (float, optional): Length or the orientation
            vector. Defaults to 0.1 * max(canvas_size) (-1 indicates default).
    """
    # Calculate all vertices (normalized)
    tangram_vertices: List[dict] = []
    for i, o in enumerate(omegas):
        tangram: Dict[str, List[np.ndarray]] = dict()
        for piece_id in PIECE_SHAPE_TEMPLATES.keys():
            id_index = int(piece_id) - 1
            theta: float = o[id_index][2] * np.pi
            R_theta: np.matrix = rotation_matrix_2d_from_theta(theta)
            piece: List[np.ndarray] = []
            for vertex in PIECE_SHAPE_TEMPLATES[piece_id]:
                vertex: np.ndarray = np.array(vertex) / TANGRAM_DATA_MAX # type: ignore
                vertex = np.asarray(R_theta * np.matrix(vertex).T).squeeze()
                vertex = vertex + o[id_index][:2]
                piece += [vertex]
            tangram[piece_id] = piece
        tangram_vertices += [tangram]
    # Find maximum canvas size
    max_width: float = tangram_vertices[0]['1'][0][0]
    min_width: float = tangram_vertices[0]['1'][0][0]
    max_height: float = tangram_vertices[0]['1'][0][1]
    min_height: float = tangram_vertices[0]['1'][0][1]
    for _, tangram in enumerate(tangram_vertices):
        for piece_id in tangram.keys():
            for vertex in tangram[piece_id]:
                if vertex[0] > max_width:
                    max_width = vertex[0]
                elif vertex[0] < min_width:
                    min_width = vertex[0]
                if vertex[1] > max_height:
                    max_height = vertex[1]
                elif vertex[1] < min_height:
                    min_height = vertex[1]
    # De-Normalization of vertices
    #coef: float = min([canvas_length / (max_width-min_width),
    #                   canvas_length / (max_height-min_height)])
    coef: float = canvas_length / max([(max_width-min_width),(max_height-min_height)])
    for i, tangram in enumerate(tangram_vertices):
        for piece_id in tangram.keys():
            for j, _ in enumerate(tangram[piece_id]):
                tangram_vertices[i][piece_id][j] *= coef
    # Draw
    frames = []
    origin: Tuple[int, int] = (int(-min_width * coef), int(-min_height * coef))
    canvas_size: Tuple[int, int] = (int((max_height-min_height) * coef),
                                    int((max_width-min_width) * coef))
    vec_len: float = 0.1 * canvas_length
    basis_vector: np.matrix = np.matrix([0.0, 1.0]).T * vec_len
    for i, tangram in enumerate(tangram_vertices):
        omega: np.ndarray = np.concatenate([omegas[i][:, :2], omegas[i][:, 2:]], axis=-1)
        assert(omega.shape[0] == 7 and omega.shape[1] == 3)
        img = draw_one_tangram(tangram=tangram,
                               omega=omega,
                               scale=coef,
                               origin=origin,
                               canvas_size=canvas_size,
                               thickness=thickness,
                               edge_color=edge_color,
                               fill_color=fill_color,
                               background_color=background_color,
                               position_point_color=position_point_color,
                               orientation_vector_color=orientation_vector_color,
                               orientation_vector_length=0)
        frames += [img]
    return frames


def draw_one_tangram(tangram: Dict[str, List[np.ndarray]],
                     omega: np.ndarray,
                     scale: float,
                     origin: Tuple[int, int],
                     canvas_size: Tuple[int, int],
                     thickness: int = 2,
                     edge_color: Tuple[int, int, int] = (0x72, 0x42, 0x57),
                     fill_color: Tuple[int, int, int] = (0x98, 0x58, 0x76),
                     background_color: Tuple[int, int, int] = (0xff, 0xff, 0xff),
                     position_point_color: Tuple[int, int, int] = (0x53, 0xd0, 0x52),
                     orientation_vector_color: Tuple[int, int, int] = (0x53, 0xd0, 0x52),
                     orientation_vector_length: float = -1.0
                     ) -> np.ndarray:
    # TODO: update
    """
    Args:
        tangram (Dict[str, List[np.ndarray]]): Vertices of each piece of a tangram.
        omega (np.ndarray): State of pieces in a tangram (positions & orientations).
        scale (float): __description__
        origin (Tuple[int, int]): __description__
        canvas_size (Tuple[int, int]): Canvas size (shape).
        thickness (int): Thickness of edges and orientation vector. And, the radius
            of a position point of a tangram piece.
        edge_color (Tuple[int, int, int]): Color of shapes' edges
        fill_color (Tuple[int, int, int]): Color to fill each piece of tangram
        background_color (Tuple[int, int, int]): Color of the canvas background
        position_point_color (Tuple[int, int, int], optional): The color to indicate
            positions of tangram pieces. Defaults to (0x53, 0xd0, 0x52).
        orientation_vector_color (Tuple[int, int, int], optional): The color to
            indicate orientation vector of tangram pieces.
            Defaults to (0x53, 0xd0, 0x52).
        orientation_vector_length (float, optional): Length or the orientation
            vector. Defaults to 0.1 * max(canvas_size) (-1 indicates default).

    Returns:
        np.ndarray: The image
    """
    img: np.ndarray = np.ones(canvas_size + (3,)) * np.asarray(background_color)
    vec_len: float = 0.05 * max(canvas_size)
    if orientation_vector_length >= 0:
        vec_len = orientation_vector_length
    basis_vector: np.matrix = np.matrix([0.0, 1.0]).T * vec_len
    for piece_id in tangram.keys():
        id_index = int(piece_id) - 1
        draw_piece(img=img,
                   piece_id=piece_id,
                   omega=omega[id_index],
                   scale=scale,
                   origin=origin,
                   thickness=thickness,
                   edge_color=edge_color,
                   fill_color=fill_color,
                   position_point_color=position_point_color,
                   orientation_vector_color=orientation_vector_color,
                   orientation_vector_length=orientation_vector_length)
    return img.astype(np.uint8)


def draw_piece(img: np.ndarray,
               piece_id: str,
               omega: np.ndarray,
               scale: float,
               origin: Tuple[int, int],
               thickness: int = 2,
               edge_color: Tuple[int, int, int] = (0x72, 0x42, 0x57),
               fill_color: Tuple[int, int, int] = (0x98, 0x58, 0x76),
               position_point_color: Tuple[int, int, int] = (0x53, 0xd0, 0x52),
               orientation_vector_color: Tuple[int, int, int] = (0x53, 0xd0, 0x52),
               orientation_vector_length: float = -1.0
               ) -> None:
    # TODO: update
    """
    Args:
        img (np.ndarray): _description_
        omega (np.ndarray): Normalized omega of ONE piece (3,)
        scale (float): _description_
        origin (Tuple[int, int]): _description_
    """
    vec_len: float = 0.05 * max(img.shape)
    if orientation_vector_length >= 0:
        vec_len = orientation_vector_length
    basis_vector: np.matrix = np.matrix([0.0, 1.0]).T * vec_len
    # De-Normalization
    position: np.ndarray = omega.copy()[:2] * scale + np.asarray(origin)
    orientation: float = omega[2] * np.pi
    # Transformation
    vertices: List[np.ndarray] = []
    R_theta: np.matrix = rotation_matrix_2d_from_theta(orientation)
    for v in PIECE_SHAPE_TEMPLATES[piece_id]:
        v: np.ndarray = np.array(v) * scale / TANGRAM_DATA_MAX # type: ignore
        v = np.asarray(R_theta * np.matrix(v).T).squeeze()
        v = v + position
        vertices += [v]
    # Outline & fill
    pts:np.ndarray = np.asarray(vertices).astype(int)
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
                center=position.astype(int),
                radius=thickness,
                color=position_point_color,
                thickness=-1)
    # Orientation
    delta: np.matrix = R_theta * basis_vector
    p1 = (np.array(position)).astype(int)
    p2 = np.array(p1 + delta.T).reshape((2,)).astype(int)
    cv2.arrowedLine(img=img,
                    pt1=p1, pt2=p2,
                    color=orientation_vector_color,
                    thickness=thickness,
                    line_type=cv2.LINE_AA,
                    shift=0,
                    tipLength=0.2)
    pass