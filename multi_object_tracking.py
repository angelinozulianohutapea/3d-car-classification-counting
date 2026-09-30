"""
Perform multi object tracking on a mp4 video with a pytorch neural network model (.pth)
Instead of only performing detections, we try to determine objects position over time.
"""

import os
from argparse import ArgumentParser
from time import time
import numpy as np
import cv2
from tqdm import tqdm

from utils import (
    draw_fps,
    draw_frame_id,
    draw_label,
    COLORS,
    get_position_ground,
    make_euler,
    project,
)
from object_detector import ObjectDetector
from kalman_filter import KalmanFilter, TIME_INTERVAL
from vehicle_shape import VehicleShape, draw_oriented_box

# ----------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------
DEBUG = False  # print log per frame (True = lambat)

MAX_LOST_FRAMES = 30  # objek dihapus setelah sekian frame tanpa deteksi
MIN_HITS_TO_DISPLAY = 3  # objek baru baru digambar setelah sekian kali terdeteksi
MAX_LOST_TO_DISPLAY = 5  # objek yang hilang > sekian frame tidak digambar
OUTPUT_DIR = "video_tracking_carla"

to_world_from_camera = np.identity(4)
pitch_matrix = make_euler(-9.80 * np.pi / 180, 0)
to_world_from_camera[:3, :3] = pitch_matrix
to_world_from_camera[:3, 3] = np.array([0, 8.30, 0])

K = np.array([[770.00, 0.0, 640.0], [0.0, 770.00, 360.0], [0.0, 0.0, 1.0]])


def log(*args):
    if DEBUG:
        print(*args)


class Object:
    """
    Object class with its current state (position, width/height/length),
    age since creation, Kalman Filter, and adaptive vehicle shape (class, dims, yaw).
    We also have an unassociated counter to be robust to misdetections
    """

    def __init__(self, id, det, dt=TIME_INTERVAL):
        self.id = id  # ID internal (untuk logika tracking)
        self.display_id = None  # nomor urut untuk tampilan, diberikan saat pertama kali visible
        self.age = 0
        self.hits = 1
        self.unassociated_counter = 0
        self.kf = KalmanFilter(
            nx=6,
            nz=5,
            first_measurement=np.asarray(
                [det["x"], det["y"], det["w"], det["h"], det["l"]], dtype=float
            ).reshape(-1),
            dt=dt,
        )
        self.shape = VehicleShape()
        self.shape.update(self.x(), self.y(), self.w(), self.h(), self.l())

    def dist(self, det):
        """Compute distance between object center and detection"""
        return np.linalg.norm(np.array([self.x() - det["x"], self.y() - det["y"]]))

    def update(self, det):
        """Update object after a match with a detection"""
        meas = np.asarray(
            [det["x"], det["y"], det["w"], det["h"], det["l"]], dtype=float
        )
        self.kf.update(meas)
        self.unassociated_counter = 0
        self.hits += 1

        # shape hanya diupdate saat ada deteksi nyata
        self.shape.update(self.x(), self.y(), self.w(), self.h(), self.l())

    def is_visible(self):
        """Objek layak digambar?"""
        return (
            self.hits >= MIN_HITS_TO_DISPLAY
            and self.unassociated_counter <= MAX_LOST_TO_DISPLAY
        )

    def x(self):
        return self.kf.estimate[0, 0]

    def y(self):
        return self.kf.estimate[1, 0]

    def w(self):
        return self.kf.estimate[3, 0]

    def h(self):
        return self.kf.estimate[4, 0]

    def l(self):
        return self.kf.estimate[5, 0]

    def get_speed(self):
        return self.kf.estimate[2, 0]

    def get_position(self):
        return self.kf.estimate[0:2, 0]


class Tracking:
    """
    Tracking class: responsible to create/kill objects,
    and to match current detections with previous Tracking state
    """

    def __init__(self, dt=TIME_INTERVAL):
        self.objects = []
        self.last_id = -1
        self.next_display_id = 1  # nomor tampilan mulai dari 1
        self.dt = dt

    def add(self, det):
        """Add a new object"""
        obj = Object(self.last_id + 1, det, dt=self.dt)
        self.objects.append(obj)
        self.last_id += 1

    def kill(self, id):
        """Kill object with id"""
        self.objects = [obj for obj in self.objects if obj.id != id]

    def assign_display_ids(self):
        """Beri nomor urut ke objek yang baru pertama kali layak digambar."""
        newly_visible = [
            obj for obj in self.objects if obj.display_id is None and obj.is_visible()
        ]
        # kalau beberapa muncul di frame yang sama, urutkan dari yang paling dekat kamera
        newly_visible.sort(key=lambda o: o.y())
        for obj in newly_visible:
            obj.display_id = self.next_display_id
            self.next_display_id += 1

    def hungarian(self, detections):
        """Compute distances between detection bboxes and tracking objects"""
        distances = np.zeros((len(self.objects), len(detections)))
        for i, obj in enumerate(self.objects):
            for j, detection in enumerate(detections):
                distances[i, j] = obj.dist(detection)

        return distances

    def update(self, detections):
        """
        Update tracking with latest detections.

        - Object dipertahankan sampai MAX_LOST_FRAMES jika detection hilang.
        - Gate matching bergantung pada jarak (objek jauh lebih noisy).
        - Object yang baru kehilangan detection mendapat toleransi lebih besar.
        - Detection yang tidak cocok dibuat sebagai object baru.
        """

        # 1. Predict semua object
        for obj in self.objects:
            obj.kf.predict()

        # 2. Distance object <-> detection
        impossible = 1000
        distances = self.hungarian(detections)

        # 3. Jika belum ada object
        if len(distances) == 0:
            for detection in detections:
                log(f"Create new object with det {detection}")
                self.add(detection)
            return

        # 4. Status matching
        match_objects = [False] * len(self.objects)
        match_detections = [False] * len(detections)

        # 5. Matching (greedy berdasarkan jarak terkecil)
        if distances.shape[1] > 0:

            while True:
                match_idx = np.unravel_index(np.argmin(distances), distances.shape)
                i = int(match_idx[0])
                j = int(match_idx[1])

                distance = distances[i, j]

                if distance >= impossible:
                    break

                obj = self.objects[i]

                # Gate adaptif: makin jauh, makin longgar
                depth = max(obj.y(), 0.0)

                if obj.unassociated_counter > 0:
                    max_distance = 4.0 + 0.05 * depth
                    max_x_distance = 2.0 + 0.01 * depth
                else:
                    max_distance = 2.0 + 0.05 * depth
                    max_x_distance = 1.0 + 0.01 * depth

                if distance > max_distance:
                    distances[i, j] = impossible
                    continue

                x_difference = abs(detections[j]["x"] - obj.x())

                if x_difference > max_x_distance:
                    distances[i, j] = impossible
                    continue

                # MATCH BERHASIL
                log(
                    f"Match object {obj.id} with detection {j} "
                    f"(distance={distance:.2f}, x_diff={x_difference:.2f})"
                )

                obj.update(detections[j])

                match_objects[i] = True
                match_detections[j] = True

                distances[i, :] = impossible
                distances[:, j] = impossible

        # 6. Detection yang tidak match -> object baru
        for j, detection in enumerate(detections):
            if not match_detections[j]:
                log(f"Create new object with det {detection}")
                self.add(detection)

        # 7. Tambah age
        for obj in self.objects:
            obj.age += 1

        # 8. Object yang tidak mendapat detection
        #    (match_objects hanya mencakup object lama; object baru ada di akhir list)
        objects = self.objects.copy()

        for i, matched in enumerate(match_objects):
            if matched:
                continue

            obj = objects[i]
            position = obj.get_position()
            obj.unassociated_counter += 1

            log(f"Object {obj.id} not detected for {obj.unassociated_counter} frames")

            if position[0] < -11 or position[0] >= 11:
                log(
                    f"Kill object {obj.id} (age {obj.age}) "
                    f"out of valid area at {position}"
                )
                self.kill(obj.id)

            elif obj.unassociated_counter >= MAX_LOST_FRAMES:
                log(
                    f"Kill object {obj.id} (age {obj.age}) at {position} "
                    f"after {obj.unassociated_counter} unassociated frames"
                )
                self.kill(obj.id)

        # 9. Hapus object yang terlalu jauh
        for obj in self.objects.copy():
            if obj.y() > 40:
                log(f"Kill object {obj.id} because y={obj.y():.2f} > 40")
                self.kill(obj.id)

        # 10. Kill conflicts (pertahankan object yang lebih tua)
        objects = self.objects.copy()

        for i, obj_1 in enumerate(objects):
            for j in range(i + 1, len(objects)):
                obj_2 = objects[j]

                distance_squared = (obj_1.x() - obj_2.x()) ** 2 + (
                    obj_1.y() - obj_2.y()
                ) ** 2

                if distance_squared < 9:
                    log("Conflict", obj_1.id, obj_2.id)

                    if obj_1.age < obj_2.age:
                        self.kill(obj_1.id)
                    else:
                        self.kill(obj_2.id)

    def __str__(self) -> str:
        if len(self.objects) == 0:
            return "Tracking is empty"

        s = "Tracking contains:\n"
        s += "------------------------------------------------------\n"
        s += "   Id   |   X   |   Y   |   W   |   H   |   L   |  Age  |  US  | Speed |\n"
        s += "------------------------------------------------------\n"
        for obj in self.objects:
            s += f"{obj.id:5}   | {obj.x():.1f}  | {obj.y():.1f}  |"
            s += f" {obj.w():.1f}  | {obj.h():.1f}  |  {obj.l():.1f}  |"
            s += f"{obj.age}  |  {obj.unassociated_counter}  |  {obj.get_speed():.1f}\n"
        return s

    def display(self, frame):
        """
        Draw each object with adaptive oriented 3D bounding box,
        and a label (ID, class, speed) above the box.
        """
        self.assign_display_ids()
        to_camera_from_world = np.linalg.inv(to_world_from_camera)

        for obj in self.objects:
            if not obj.is_visible() or obj.display_id is None:
                continue

            color = COLORS[obj.display_id % len(COLORS)]
            dims = obj.shape.dims()  # (w, h, l) hasil template + ukuran terukur

            frame = draw_oriented_box(
                frame,
                obj.x(),
                obj.y(),
                dims,
                obj.shape.yaw,
                to_camera_from_world,
                K,
                color,
            )

            # titik tengah atap kotak -> label ditaruh di atasnya
            top = project(
                np.array([obj.x(), dims[1], obj.y()]), to_camera_from_world, K
            )
            tx, ty = top[0], frame.shape[0] - top[1]

            # objek jauh = font lebih kecil
            scale = float(np.clip(0.8 - obj.y() / 80.0, 0.45, 0.8))

            title = f"ID {obj.display_id} ({obj.shape.cls})"
            speed = f"{abs(obj.get_speed() * 3.6):.0f} km/h"

            draw_label(frame, [title, speed], tx, ty - 6, color, scale=scale)

        return frame


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-object tracking")
    parser.add_argument("video", type=str, help="Video")
    parser.add_argument("model", type=str, help="Pytorch model for bbox cars detection")
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Threshold to keep an object"
    )
    parser.add_argument(
        "-f", type=int, default=np.inf, help="Pause viz at specified frame"
    )
    args = parser.parse_args()

    object_detector = ObjectDetector(args.model, args.conf)

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("Error opening video stream or file")

    # Jumlah frame dan FPS asli video (dipakai sebagai dt Kalman Filter)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / video_fps if video_fps and video_fps > 1 else TIME_INTERVAL
    print(f"Video FPS: {video_fps:.1f} -> dt = {dt:.4f} s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prev_time = time()

    tracking = Tracking(dt=dt)
    PLAY = False

    for idx in tqdm(range(n_frames)):
        if not cap.isOpened():
            break

        ret, frame = cap.read()
        if not ret:
            break

        log(f"\n#################### Frame {idx} #####################")

        bboxs = object_detector.detect(frame)

        world_bboxs = []

        for box in bboxs:
            position_on_ground = get_position_ground(
                box["x"], box["y"], K, to_world_from_camera, 720
            )

            if abs(position_on_ground[0]) > 11:
                continue

            depth = np.linalg.norm(position_on_ground - to_world_from_camera[:3, 3])

            if box["w"] * depth < 1:
                continue

            if box["h"] * depth < 1 or box["h"] * depth > 5:
                continue

            if box["l"] * depth < 3:
                continue

            if box["h"] > box["l"]:
                continue

            world_bbox = {
                "x": position_on_ground[0],
                "y": position_on_ground[2],
                "w": min(box["w"] * depth, 2.5),
                "h": box["h"] * depth,
                "l": box["l"] * depth,
            }

            world_bboxs.append(world_bbox)

        tracking.update(world_bboxs)
        log(tracking)

        frame = tracking.display(frame)

        fps = 1 / (time() - prev_time)
        prev_time = time()
        draw_fps(frame, fps)
        draw_frame_id(frame, idx)

        cv2.imshow("Detection", frame)
        k = cv2.waitKey(10 if ((idx < args.f < np.inf) or PLAY) else 0)

        if k == 27:
            break

        if k == 32:  # SPACE
            PLAY = not PLAY

        cv2.imwrite(f"{OUTPUT_DIR}/image_{idx:04d}.png", frame)