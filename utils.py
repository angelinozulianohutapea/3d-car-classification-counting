"""
Define a bunch of utils class and functions related to geometry and opencv.
"""

from itertools import combinations

import cv2
import numpy as np


def project(point_in_world, to_camera_from_world, K):
    """
    Given the camera intrinsic and extrinsic calibration matrices,
    projet a 3D point in world coordinates to the image
    """

    point_in_world = point_in_world.reshape(3, 1)
    point_in_camera = to_camera_from_world[:3, :3].dot(
        point_in_world
    ) + to_camera_from_world[:3, 3].reshape(3, 1)

    point_in_camera /= abs(point_in_camera[2])

    projection = K.dot(point_in_camera)
    projection = projection.flatten()

    return projection[:2]


def make_euler(angle, axis):
    """Define euler matrix with some angle around a specified axis"""

    if axis < 0:
        raise ValueError
    if axis > 3:
        raise ValueError

    if axis == 0:  # Pitch (axis = X)
        pitch_matrix = np.zeros((3, 3))
        pitch_matrix[0, 0] = 1
        pitch_matrix[1, 1] = np.cos(angle)
        pitch_matrix[1, 2] = np.sin(angle)
        pitch_matrix[2, 1] = -np.sin(angle)
        pitch_matrix[2, 2] = np.cos(angle)
        return pitch_matrix

    if axis == 1:  # Yaw (axis = Y)
        yaw_matrix = np.zeros((3, 3))
        yaw_matrix[0, 0] = np.cos(angle)
        yaw_matrix[0, 2] = np.sin(angle)
        yaw_matrix[1, 1] = 1
        yaw_matrix[2, 0] = -np.sin(angle)
        yaw_matrix[2, 2] = np.cos(angle)
        return yaw_matrix

    return False


def get_position_ground(x, y, K, to_world_from_camera, img_height):
    """
    Find position in world on the driveway (altitude = 0)
    by ray tracing a pixel point [x, y]
    """

    point_in_camera = np.linalg.inv(K).dot(
        np.array([x, img_height - y, 1]).reshape(3, 1)
    )
    point_in_camera = point_in_camera.reshape(3, 1)
    # Find intersection with Y = 0

    point_in_world = to_world_from_camera[:3, :3].dot(
        point_in_camera
    ) + to_world_from_camera[:3, 3].reshape(3, 1)

    camera_pos_in_world = to_world_from_camera[:3, 3].reshape(3, 1)

    direction = point_in_world - camera_pos_in_world
    direction /= np.linalg.norm(direction)

    lambda_ = -camera_pos_in_world[1] / direction[1]

    position_on_ground = (camera_pos_in_world + lambda_ * direction).flatten()

    return position_on_ground


def draw_fps(frame, fps):
    """Draw fps to demonstrate performance"""
    cv2.putText(
        frame,
        f"{int(fps)} fps",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )


def draw_frame_id(frame, frame_id):
    """Draw frame id to demonstrate performance"""
    cv2.putText(
        frame,
        f"Frame {frame_id}",
        (frame.shape[1] - 150, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )


def draw_label(frame, lines, x, y, color, scale=0.6, thickness=1):
    """
    Gambar label (beberapa baris) dengan latar berwarna.
    Bagian bawah label berada di y, dan label dipusatkan di x.
    Posisi dijaga tetap di dalam frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 4

    sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
    w = max(s[0] for s in sizes) + 2 * pad
    line_h = max(s[1] for s in sizes) + 6
    h = line_h * len(lines) + pad

    x0 = int(max(0, min(x - w / 2, frame.shape[1] - w)))
    y0 = int(max(0, min(y - h, frame.shape[0] - h)))

    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), color, thickness=-1)

    # teks hitam kalau warna latar terang, putih kalau gelap
    b, g, r = color
    text_color = (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 140 else (255, 255, 255)

    for i, t in enumerate(lines):
        cv2.putText(
            frame,
            t,
            (x0 + pad, y0 + (i + 1) * line_h - 2),
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )


# Define a bunch of RGB colors (used in tracking)
COLORS = [
    (0, 0, 128),
    (0, 0, 255),
    (0, 128, 0),
    (0, 128, 128),
    (0, 128, 255),
    (0, 255, 0),
    (0, 255, 128),
    (0, 255, 255),
    (128, 0, 0),
    (128, 0, 128),
    (128, 0, 255),
    (128, 128, 0),
    (128, 128, 128),
    (128, 128, 255),
    (128, 255, 0),
    (128, 255, 128),
    (128, 255, 255),
    (255, 0, 0),
    (255, 0, 128),
    (255, 0, 255),
    (255, 128, 0),
    (255, 128, 128),
    (255, 128, 255),
    (255, 255, 0),
    (255, 255, 128),
    (255, 255, 255),
]


class Box:
    """
    Define 3D bounding box (sejajar sumbu, tanpa yaw)
    """

    def __init__(self, x, z, w, h, l):
        self.x = x
        self.z = z
        self.w = w
        self.h = h
        self.l = l

    def project(self, img, to_world_from_camera, K, color=(255, 0, 255)):
        """
        Project 3D bounding box to the image img
        """
        to_camera_from_world = np.linalg.inv(to_world_from_camera)

        # 8 sudut: kunci = (sisi x, atas?, sisi z), masing-masing 0 atau 1
        pts = {}
        for sx in (0, 1):
            for up in (0, 1):
                for sz in (0, 1):
                    world = np.array(
                        [
                            self.x + (sx - 0.5) * self.w,
                            self.h * up,
                            self.z + (sz - 0.5) * self.l,
                        ]
                    )
                    p = project(world, to_camera_from_world, K)
                    p[1] = img.shape[0] - p[1]
                    pts[(sx, up, sz)] = tuple(p.astype(int))

        # 12 rusuk: pasangan sudut yang hanya beda di satu koordinat
        for a, b in combinations(pts.keys(), 2):
            if sum(i != j for i, j in zip(a, b)) == 1:
                cv2.line(img, pts[a], pts[b], color=color, thickness=3)

        # Overlay merah transparan di seluruh kotak
        image_overlay = img.copy()
        hull = cv2.convexHull(np.array(list(pts.values()), dtype=int)).reshape(1, -1, 2)
        cv2.fillPoly(image_overlay, pts=hull, color=(0, 0, 255))

        alpha = 0.3
        img = cv2.addWeighted(img, 1 - alpha, image_overlay, alpha, 0)

        return img