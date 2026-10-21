]633;E;echo "=== LS ===";9fcc772e-1df4-4b5d-8565-40efeb2c08a5]633;C=== LS ===
13834567_1280x720.mp4
13834567_frame700_detection.jpg
13834567_tracking_result.mp4
analisis_cache.py
annotation
bandingkan_ukuran.py
calibrate_tuner.py
camera_calibration.py
carla_3d_bbox.jpg
carla_dataset
carla_detection.jpg
carla_results
cek_base
cek_fp.py
cek_ft
centernet-3d-bbox.ipynb
centernet-3d-bbox.pth
centernet-ft10.pth
centernet-ft.pth
counting_base03.csv
counting_log.csv
create_test_video.py
det_ft.pkl
det_low.pkl
det.pkl
dlr_detection.jpg
dlr_detection_resize.jpg
dlr_frame.jpg
DLR_HT_060000_060100.mp4
dlr_test_frames
finetune_heatmap.py
hasil_3d_bbox.jpg
hasil_base_03.txt
hasil_dua_tahap.txt
hasil_ft.txt
istock_frame50_detection.jpg
istockphoto-981340772-640_adpp_is.mp4
istock_test_detection.jpg
istock_test_frame.jpg
kalman_filter.py
kode_untuk_readme.txt
label_points.py
labels.json
lembar_1.png
lembar_2.png
lembar_3.png
lembar_hitungan.py
make_tracking_video.py
multi_object_detection.py
multi_object_tracking.py
object_detector.py
periph.jpg
__pycache__
README.md
res
test_1280_frame235_bbox.jpg
test_1280_frame235.jpg
test_3d_bbox.py
test_all_carla.py
test_carla_3d.py
test_carla_bbox.py
test_carla.py
test_dlr_bbox.py
test_dlr_detection.py
test_dlr_multiple.py
test_dlr_resize.py
test_model.py
test_video_frame.py
test_video.mp4
tracking_log.txt
tracking_result.mp4
utils.py
vehicle_shape.py
venv
video_tracking_carla
visualize_carla_annotation.py
=== HELP ===
Command 'python' not found, did you mean:
  command 'python3' from deb python3
  command 'python' from deb python-is-python3

=== multi_object_tracking.py ===
"""
Perform multi object tracking on a mp4 video with a pytorch neural network model (.pth)
Instead of only performing detections, we try to determine objects position over time.

Tambahan dibanding versi sebelumnya (semua default = perilaku lama, kecuali --nms):
  --nms pool|legacy     puncak heatmap: max-pool 3x3 (default) atau cara lama
  --cache FILE          simpan/muat deteksi mentah (detektor dijalankan sekali saja)
  --cache-conf X        ambang terendah saat membuat cache (default 0.2)
  --max-width X         batas lebar kendaraan (m); 0 = nonaktif (default 2.5)
  --conflict-dist X     jarak (m) di bawah mana dua track dianggap duplikat (default 3.0)
  --no-save             jangan tulis PNG tiap frame (jauh lebih cepat)
  --conf-low X          asosiasi dua tahap ala ByteTrack: deteksi skor X..--conf hanya
                        dipakai untuk melanjutkan track yang sudah ada, tidak membuat track baru
                        (Zhang dkk., 2022, arXiv:2110.06864). Default: nonaktif.
"""

import csv
import os
import pickle
from argparse import ArgumentParser
from time import time
import numpy as np
from scipy.optimize import linear_sum_assignment
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

# Garis virtual untuk counting (jarak dari kamera, meter, searah sumbu jalan).
# Atur dengan --line-y. Garis membentang selebar jalur valid (|x| <= 11 m).
LINE_Y_DEFAULT = 15.0
LINE_HALF_WIDTH = 11.0
COUNT_CLASSES = ("car", "van", "truck")
COUNT_LOG = "counting_log.csv"

# Lebar minimum (meter) agar deteksi dipakai. Sebelumnya 1.0; dilonggarkan
# karena estimasi lebar dari model CARLA sering meleset di video nyata.
MIN_WIDTH_M = 0.6

DEFAULT_MAX_WIDTH_M = 2.5
DEFAULT_CONFLICT_DIST_M = 3.0

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
        self.counted = False  # sudah dihitung saat melewati garis virtual?
        self.crossed = False  # sudah pernah melewati garis (walau belum layak tampil)
        self.prev_y = None  # posisi y sebelumnya, untuk deteksi lintasan garis
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

    def __init__(self, dt=TIME_INTERVAL, conflict_dist=DEFAULT_CONFLICT_DIST_M):
        self.objects = []
        self.last_id = -1
        self.next_display_id = 1  # nomor tampilan mulai dari 1
        self.dt = dt
        self.conflict_dist_sq = conflict_dist ** 2
        self.counts = {c: 0 for c in COUNT_CLASSES}
        self.events = []  # (frame, display_id, kelas, ...)
        self.history = {}  # display_id -> ringkasan hidup track

    def count_crossings(self, line_y, frame_idx):
        """
        Hitung kendaraan yang melewati garis virtual (tanpa arah).
        Satu track dihitung paling banyak sekali, dengan kelas hasil voting
        pada saat melintas. Hanya track yang sudah layak tampil yang dihitung,
        supaya deteksi palsu sesaat tidak masuk hitungan.
        """
        self.assign_display_ids()
        for obj in self.objects:
            y = obj.y()
            prev = obj.prev_y
            if obj.display_id is not None:
                rec = self.history.setdefault(
                    obj.display_id,
                    {"first": frame_idx, "y_first": y, "y_min": y, "y_max": y, "x_first": obj.x()},
                )
                rec["last"] = frame_idx
                rec["y_last"] = y
                rec["y_min"] = min(rec["y_min"], y)
                rec["y_max"] = max(rec["y_max"], y)
                rec["cls"] = obj.shape.cls
                rec["counted"] = obj.counted
                rec["lost"] = obj.unassociated_counter
            obj.prev_y = y
            # tanda (y - line_y) berubah = melintasi garis dari arah mana pun.
            # Lintasan dicatat walau track sedang tidak terlihat, supaya tidak hilang.
            if (
                prev is not None
                and not obj.crossed
                and (prev - line_y) * (y - line_y) <= 0
                and prev != y
            ):
                obj.crossed = True

            # dihitung setelah track layak tampil (bukan deteksi palsu sesaat)
            if obj.crossed and not obj.counted and obj.display_id is not None:
                obj.counted = True
                cls = obj.shape.cls
                self.counts[cls] += 1
                med = np.median(np.array(obj.shape.dim_buf), axis=0)
                votes = obj.shape.votes
                self.events.append(
                    (
                        frame_idx,
                        obj.display_id,
                        cls,
                        round(float(med[2]), 2),  # panjang terukur (median)
                        round(float(med[0]), 2),  # lebar terukur
                        round(float(med[1]), 2),  # tinggi terukur
                        round(float(obj.x()), 2),  # posisi lateral saat melintas
                        votes["car"],
                        votes["van"],
                        votes["truck"],
                    )
                )
                log(f"COUNT frame {frame_idx}: ID {obj.display_id} ({cls})")

    def total_count(self):
        return sum(self.counts.values())

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

    def distance_matrix(self, detections):
        """Hitung jarak antara setiap objek terlacak dan setiap deteksi"""
        distances = np.zeros((len(self.objects), len(detections)))
        for i, obj in enumerate(self.objects):
            for j, detection in enumerate(detections):
                distances[i, j] = obj.dist(detection)

        return distances

    def _match(self, obj_ids, dets):
        """
        Hungarian (SciPy) dengan gate adaptif yang sama seperti sebelumnya.
        obj_ids: indeks di self.objects; dets: daftar deteksi.
        Mengembalikan daftar pasangan (indeks objek, indeks deteksi).
        """
        if not obj_ids or not dets:
            return []
        impossible = 1000.0
        cost = np.full((len(obj_ids), len(dets)), impossible)
        for a, i in enumerate(obj_ids):
            obj = self.objects[i]
            # Gate adaptif: makin jauh, makin longgar
            depth = max(obj.y(), 0.0)
            if obj.unassociated_counter > 0:
                max_distance = 4.0 + 0.05 * depth
                max_x_distance = 2.0 + 0.01 * depth
            else:
                max_distance = 2.0 + 0.05 * depth
                max_x_distance = 1.0 + 0.01 * depth
            for j, det in enumerate(dets):
                d = obj.dist(det)
                if d > max_distance:
                    continue
                if abs(det["x"] - obj.x()) > max_x_distance:
                    continue
                cost[a, j] = d
        rows, cols = linear_sum_assignment(cost)
        return [(obj_ids[a], j) for a, j in zip(rows, cols) if cost[a, j] < impossible]

    def update(self, detections, conf_high=None):
        """
        Update tracking with latest detections.

        - Object dipertahankan sampai MAX_LOST_FRAMES jika detection hilang.
        - Gate matching bergantung pada jarak (objek jauh lebih noisy).
        - Object yang baru kehilangan detection mendapat toleransi lebih besar.
        - Detection yang tidak cocok dibuat sebagai object baru.

        conf_high=None: perilaku lama (satu tahap, semua deteksi setara).
        conf_high=X: asosiasi dua tahap ala ByteTrack (Zhang dkk., 2022,
        arXiv:2110.06864). Tahap 1: semua track dicocokkan dengan deteksi skor >= X.
        Tahap 2: track yang belum cocok dicocokkan dengan deteksi skor < X.
        Track baru HANYA dibuat dari deteksi skor tinggi yang belum cocok, jadi
        deteksi lemah bisa menyambung track tetapi tidak bisa memunculkan track.
        """

        # 1. Predict semua object
        for obj in self.objects:
            obj.kf.predict()

        # 2. Pisahkan deteksi skor tinggi dan rendah
        if conf_high is None:
            high_dets, low_dets = list(detections), []
        else:
            high_dets = [d for d in detections if d.get("score", 1.0) >= conf_high]
            low_dets = [d for d in detections if d.get("score", 1.0) < conf_high]

        # 3. Jika belum ada object
        if len(self.objects) == 0:
            for detection in high_dets:
                log(f"Create new object with det {detection}")
                self.add(detection)
            return

        # 4. Status matching (match_objects hanya mencakup object lama)
        match_objects = [False] * len(self.objects)
        match_high = [False] * len(high_dets)

        # 5a. Tahap 1: semua object <-> deteksi skor tinggi
        for i, j in self._match(list(range(len(self.objects))), high_dets):
            obj = self.objects[i]
            log(f"Match object {obj.id} with high detection {j}")
            obj.update(high_dets[j])
            match_objects[i] = True
            match_high[j] = True

        # 5b. Tahap 2: object yang belum cocok <-> deteksi skor rendah
        rest = [i for i, m in enumerate(match_objects) if not m]
        for i, j in self._match(rest, low_dets):
            obj = self.objects[i]
            log(f"Match object {obj.id} with low detection {j}")
            obj.update(low_dets[j])
            match_objects[i] = True

        # 6. Deteksi skor tinggi yang tidak match -> object baru
        for j, detection in enumerate(high_dets):
            if not match_high[j]:
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

                if distance_squared < self.conflict_dist_sq:
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


def draw_counting_line(frame, line_y):
    """Gambar garis virtual di bidang jalan pada kedalaman line_y."""
    to_camera_from_world = np.linalg.inv(to_world_from_camera)
    h = frame.shape[0]
    pts = []
    for x in (-LINE_HALF_WIDTH, LINE_HALF_WIDTH):
        pix = project(np.array([x, 0.0, line_y]), to_camera_from_world, K)
        pts.append((int(pix[0]), int(h - pix[1])))
    cv2.line(frame, pts[0], pts[1], (0, 0, 255), 3)
    return frame


def draw_counts(frame, counts):
    """Panel hitungan per kelas di pojok kiri bawah."""
    lines = [f"Total: {sum(counts.values())}"]
    lines += [f"{c}: {counts[c]}" for c in COUNT_CLASSES]
    h = frame.shape[0]
    box_h = 26 * len(lines) + 10
    cv2.rectangle(frame, (10, h - box_h - 10), (190, h - 10), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (18, h - box_h + 18 + 26 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
    return frame


def draw_debug(frame, marks):
    """Tandai deteksi mentah: hijau = lolos, kuning = lolos skor rendah, merah = ditolak."""
    for x, y, reason, low in marks:
        pt = (int(x), int(y))
        if reason is None:
            # hijau = lolos, kuning = lolos tetapi skor rendah (hanya menyambung track)
            cv2.circle(frame, pt, 6, (0, 255, 255) if low else (0, 255, 0), -1)
        else:
            cv2.circle(frame, pt, 6, (0, 0, 255), -1)
            cv2.putText(frame, reason, (pt[0] + 8, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame


def save_counting_log(tracking, path=COUNT_LOG, line_y=LINE_Y_DEFAULT):
    """Simpan tiap kejadian counting dan cetak ringkasannya."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame", "id", "kelas", "panjang_m", "lebar_m", "tinggi_m", "x_m",
             "suara_car", "suara_van", "suara_truck"]
        )
        writer.writerows(tracking.events)
    print(f"\nHasil counting (disimpan di {path}):")
    for c in COUNT_CLASSES:
        print(f"  {c:6}: {tracking.counts[c]}")
    print(f"  total : {tracking.total_count()}")
    uncounted = {k: v for k, v in tracking.history.items() if not v["counted"]}
    print(
        f"\nTrack yang tampil: {len(tracking.history)}, "
        f"dihitung: {len(tracking.history) - len(uncounted)}, "
        f"tidak dihitung: {len(uncounted)}"
    )
    if uncounted:
        print("Track tampil tapi tidak dihitung:")
        print("  id   kelas   frame (awal-akhir)   y awal -> y akhir (m)   x awal   keterangan")
        for k in sorted(uncounted):
            v = uncounted[k]
            if v["y_min"] > line_y:
                note = "tidak pernah sampai garis (selalu lebih jauh dari garis)"
            elif v["y_max"] < line_y:
                note = "lahir setelah garis (selalu lebih dekat dari garis)"
            else:
                note = "melewati garis tapi tidak terhitung (cek)"
            print(
                f"  {k:3}  {v['cls']:6}  {v['first']:5}-{v['last']:<5}"
                f"  {v['y_first']:6.1f} -> {v['y_last']:6.1f}  {v['x_first']:6.1f}   {note}"
            )
    print("\nDetail tiap kendaraan yang dihitung:")
    print("  frame   id  kelas   panjang  lebar  tinggi  x(m)  suara(car/van/truck)")
    for e in tracking.events:
        print(
            f"  {e[0]:5}  {e[1]:3}  {e[2]:6}  {e[3]:6.2f}  {e[4]:5.2f}  {e[5]:6.2f}"
            f"  {e[6]:5.1f}  {e[7]}/{e[8]}/{e[9]}"
        )


def load_detection_cache(path, nms, conf):
    """Muat cache deteksi mentah; pastikan mode NMS sama dan ambang cukup rendah."""
    with open(path, "rb") as f:
        cache = pickle.load(f)  # hanya muat file yang Anda buat sendiri
    if cache["nms"] != nms:
        raise SystemExit(
            f"Cache dibuat dengan --nms {cache['nms']}, tetapi diminta --nms {nms}. "
            "Hapus cache atau samakan opsinya."
        )
    if conf < cache["min_conf"] - 1e-9:
        raise SystemExit(
            f"--conf {conf} lebih rendah dari ambang cache ({cache['min_conf']}). "
            "Buat ulang cache dengan --cache-conf yang lebih rendah."
        )
    return cache["frames"]


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-object tracking")
    parser.add_argument("video", type=str, help="Video")
    parser.add_argument("model", type=str, help="Pytorch model for bbox cars detection")
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Threshold to keep an object"
    )
    parser.add_argument(
        "--conf-low",
        type=float,
        default=None,
        help="Aktifkan asosiasi dua tahap (ByteTrack): deteksi skor antara ini dan --conf "
        "hanya menyambung track yang ada. Harus lebih kecil dari --conf",
    )
    parser.add_argument(
        "-f", type=int, default=np.inf, help="Pause viz at specified frame"
    )
    parser.add_argument(
        "--line-y",
        type=float,
        default=LINE_Y_DEFAULT,
        help="Posisi garis virtual counting (meter dari kamera)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Tanpa jendela (untuk SSH/VS Code remote): hanya simpan frame dan hitungan",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Langsung jalan tanpa menunggu tombol (mode jendela)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Tandai semua deteksi mentah: hijau = lolos filter, merah = ditolak (dengan alasan)",
    )
    parser.add_argument(
        "--cam-height",
        type=float,
        default=None,
        help="Timpa tinggi kamera (meter) untuk percobaan kalibrasi, default dari kode (8.30)",
    )
    parser.add_argument(
        "--nms",
        choices=["pool", "legacy"],
        default="pool",
        help="Ekstraksi puncak heatmap: pool = max-pool 3x3 (CenterNet asli), legacy = cara lama",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="File cache deteksi mentah. Ada = dimuat (detektor tidak dijalankan); belum ada = dibuat",
    )
    parser.add_argument(
        "--cache-conf",
        type=float,
        default=0.2,
        help="Ambang confidence terendah saat membuat cache (default 0.2)",
    )
    parser.add_argument(
        "--max-width",
        type=float,
        default=DEFAULT_MAX_WIDTH_M,
        help="Batas atas lebar kendaraan (m); 0 = nonaktif. Default 2.5 (perilaku lama)",
    )
    parser.add_argument(
        "--conflict-dist",
        type=float,
        default=DEFAULT_CONFLICT_DIST_M,
        help="Dua track yang lebih dekat dari ini (m) dianggap duplikat. Default 3.0",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Jangan simpan PNG tiap frame (untuk eksperimen cepat)",
    )
    args = parser.parse_args()
    if args.cam_height is not None:
        to_world_from_camera[1, 3] = args.cam_height

    if args.conf_low is not None and not (0 < args.conf_low < args.conf):
        raise SystemExit("--conf-low harus > 0 dan lebih kecil dari --conf")
    # ambang terendah yang dibaca dari detektor/cache
    min_conf = args.conf_low if args.conf_low is not None else args.conf

    if args.cache and args.nms == "legacy":
        raise SystemExit(
            "--cache hanya didukung dengan --nms pool (NMS legacy bergantung pada threshold)."
        )

    # --- sumber deteksi: cache (jika ada) atau detektor ---
    cached_frames = None
    object_detector = None
    new_cache = None
    if args.cache and os.path.exists(args.cache):
        cached_frames = load_detection_cache(args.cache, args.nms, min_conf)
        print(f"Memakai cache deteksi: {args.cache} ({len(cached_frames)} frame)")
    else:
        detect_conf = args.cache_conf if args.cache else min_conf
        if args.cache and min_conf < detect_conf:
            detect_conf = min_conf
        object_detector = ObjectDetector(args.model, detect_conf, nms=args.nms)
        if args.cache:
            new_cache = {"nms": args.nms, "min_conf": detect_conf, "frames": []}

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise SystemExit("Error opening video stream or file")

    # Jumlah frame dan FPS asli video (dipakai sebagai dt Kalman Filter)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / video_fps if video_fps and video_fps > 1 else TIME_INTERVAL
    print(f"Video FPS: {video_fps:.1f} -> dt = {dt:.4f} s")
    print(
        f"Opsi: conf={args.conf} conf_low={args.conf_low} nms={args.nms} cam_height={to_world_from_camera[1, 3]} "
        f"max_width={args.max_width} conflict_dist={args.conflict_dist} line_y={args.line_y}"
    )

    if not args.no_save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    prev_time = time()

    reject_stats = {}
    dim_stats = {"lolos_w": [], "lolos_h": [], "lolos_l": [], "tolak_tinggi_h": []}

    tracking = Tracking(dt=dt, conflict_dist=args.conflict_dist)
    PLAY = args.play
    need_draw = (not args.no_show) or (not args.no_save)
    finished = True

    for idx in tqdm(range(n_frames)):
        if not cap.isOpened():
            finished = False
            break

        ret, frame = cap.read()
        if not ret:
            finished = False
            break

        log(f"\n#################### Frame {idx} #####################")

        if cached_frames is not None:
            raw = cached_frames[idx] if idx < len(cached_frames) else []
            bboxs = [b for b in raw if b["score"] >= min_conf]
        else:
            bboxs = object_detector.detect(frame)
            if new_cache is not None:
                new_cache["frames"].append(bboxs)
                bboxs = [b for b in bboxs if b["score"] >= min_conf]

        world_bboxs = []
        debug_marks = []  # (x_px, y_px, alasan ditolak atau None)

        for box in bboxs:
            position_on_ground = get_position_ground(
                box["x"], box["y"], K, to_world_from_camera, 720
            )
            depth = np.linalg.norm(position_on_ground - to_world_from_camera[:3, 3])

            reason = None
            if abs(position_on_ground[0]) > 11:
                reason = "luar jalur"
            elif box["w"] * depth < MIN_WIDTH_M:
                reason = "w kecil"
            elif box["h"] * depth < 1 or box["h"] * depth > 5:
                reason = "tinggi"
            elif box["l"] * depth < 3:
                reason = "l pendek"
            elif box["h"] > box["l"]:
                reason = "h>l"

            low = args.conf_low is not None and box["score"] < args.conf
            debug_marks.append((box["x"], box["y"], reason, low))

            if reason is None:
                dim_stats["lolos_w"].append(box["w"] * depth)
                dim_stats["lolos_h"].append(box["h"] * depth)
                dim_stats["lolos_l"].append(box["l"] * depth)
            elif reason == "tinggi":
                dim_stats["tolak_tinggi_h"].append(box["h"] * depth)

            if reason is not None:
                reject_stats[reason] = reject_stats.get(reason, 0) + 1
                continue

            width = box["w"] * depth
            if args.max_width > 0:
                width = min(width, args.max_width)

            world_bbox = {
                "x": position_on_ground[0],
                "y": position_on_ground[2],
                "w": width,
                "h": box["h"] * depth,
                "l": box["l"] * depth,
                "score": box["score"],
            }

            world_bboxs.append(world_bbox)

        tracking.update(world_bboxs, conf_high=args.conf if args.conf_low is not None else None)
        tracking.count_crossings(args.line_y, idx)
        log(tracking)

        if need_draw:
            frame = tracking.display(frame)
            frame = draw_counting_line(frame, args.line_y)
            if args.debug:
                frame = draw_debug(frame, debug_marks)
            frame = draw_counts(frame, tracking.counts)

            fps = 1 / (time() - prev_time)
            prev_time = time()
            draw_fps(frame, fps)
            draw_frame_id(frame, idx)

        if not args.no_show:
            cv2.imshow("Detection", frame)
            k = cv2.waitKey(10 if ((idx < args.f < np.inf) or PLAY) else 0)

            if k == 27:
                finished = False
                break

            if k == 32:  # SPACE
                PLAY = not PLAY

        if not args.no_save:
            cv2.imwrite(f"{OUTPUT_DIR}/image_{idx:04d}.png", frame)

    cap.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    if new_cache is not None and finished and len(new_cache["frames"]) == n_frames:
        with open(args.cache, "wb") as f:
            pickle.dump(new_cache, f)
        print(f"\nCache deteksi disimpan: {args.cache}")

    save_counting_log(tracking, line_y=args.line_y)
    if reject_stats:
        print("\nDeteksi mentah yang dibuang filter (per alasan):")
        for reason, n in sorted(reject_stats.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:12}: {n}")

    names = {
        "lolos_w": "Lebar deteksi yang lolos (m)",
        "lolos_h": "Tinggi deteksi yang lolos (m)",
        "lolos_l": "Panjang deteksi yang lolos (m)",
        "tolak_tinggi_h": "Tinggi deteksi yang ditolak karena tinggi (m)",
    }
    print("\nRingkasan dimensi (min / median / maks, jumlah):")
    for key, label in names.items():
        v = dim_stats[key]
        if v:
            print(
                f"  {label:48}: {min(v):.2f} / {np.median(v):.2f} / {max(v):.2f}  (n={len(v)})"
            )
    print("  Acuan mobil nyata: lebar ~1.7-2.0, tinggi ~1.4-1.6, panjang ~4.0-5.0 m")
=== object_detector.py ===
import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from torchvision import transforms
import numpy as np


def select_legacy(hm, threshold):
    """
    Versi lama: tekan puncak yang berjarak <= 2 sel (stride 32 -> ~64 piksel).
    Dipertahankan agar hasil lama bisa direproduksi (--nms legacy).
    """
    pred = hm > threshold
    pred_centers = np.argwhere(pred)

    for i, ci in enumerate(pred_centers):
        for j in range(i + 1, len(pred_centers)):
            cj = pred_centers[j]
            if np.linalg.norm(ci - cj) <= 2:
                score_i = hm[ci[0], ci[1]]
                score_j = hm[cj[0], cj[1]]
                if score_i > score_j:
                    hm[cj[0], cj[1]] = 0
                else:
                    hm[ci[0], ci[1]] = 0

    return hm


def select_pool(hm):
    """
    Ambil puncak lokal dengan max-pooling 3x3 (stride 1), lalu buang sel yang
    bukan maksimum di lingkungannya.

    Dasar: Zhou, Wang, Krahenbuhl (2019), "Objects as Points", arXiv:1904.07850,
    bagian 4 (peak extraction): puncak diambil dengan max-pooling 3x3, sebagai
    pengganti NMS. Dengan stride 32, radius 1 sel ~ 32 piksel, jadi dua mobil
    yang berdekatan tidak ikut saling menekan seperti pada select_legacy.
    Tidak bergantung pada threshold: threshold diterapkan di pred2box.
    """
    t = torch.from_numpy(np.ascontiguousarray(hm)).unsqueeze(0).unsqueeze(0)
    keep = F.max_pool2d(t, kernel_size=3, stride=1, padding=1) == t
    return hm * keep.squeeze(0).squeeze(0).numpy()


def select(hm, threshold, mode="pool"):
    if mode == "legacy":
        return select_legacy(hm, threshold)
    if mode == "pool":
        return select_pool(hm)
    raise ValueError(f"mode NMS tidak dikenal: {mode}")


class ObjectDetector:
    """
    Initialize ObjectDetector by loading model
    Define confidence threshold
    Then preprocess, infer, post-process
    """

    INPUT_WIDTH = 1280
    INPUT_HEIGHT = 720
    MODEL_SCALE = 32

    def __init__(self, model_pth, conf_threhsold, nms="pool"):
        self.conf = conf_threhsold
        self.nms = nms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Device: {self.device} | NMS: {self.nms}")

        # Define and load model
        self.model = centernet()
        self.model.load_state_dict(
            torch.load(model_pth, map_location=torch.device("cpu"))
        )
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def detect(self, frame):
        """
        Detect objects on frame and return bounding boxes.
        Pre-process for model input
        Extract heatmaps, and compute predicted bounding boxes
        Setiap box berisi juga "score" (nilai heatmap di puncaknya).
        """

        input_tensor = self.preprocess(frame)

        # Inference (tanpa gradient: lebih hemat memori dan sedikit lebih cepat)
        with torch.no_grad():
            hm, offset, whl = self.model(
                input_tensor.to(self.device).float().unsqueeze(0)
            )

        hm = torch.sigmoid(hm)

        hm = hm.cpu().detach().numpy().squeeze(0).squeeze(0)
        offset = offset.cpu().detach().numpy().squeeze(0)
        whl = whl.cpu().detach().numpy().squeeze(0)

        hm = select(hm, self.conf, self.nms)

        boxes = self.pred2box(hm, offset, whl, self.conf)

        return boxes

    def pred2box(self, hm, offset, regr, thresh=0.99):
        # make binding box from heatmaps
        # thresh: threshold for logits.

        # get center
        pred = hm > thresh
        pred_center = np.where(hm > thresh)

        # get regressions (urutan sama dengan np.where: row-major)
        pred_r = regr[:, pred].T
        scores = hm[pred]

        boxes = []

        pred_center = np.asarray(pred_center).T

        for center, b, score in zip(pred_center, pred_r, scores):
            offset_xy = offset[:, center[0], center[1]]
            bbox = {
                "x": (center[1] + offset_xy[0]) * self.MODEL_SCALE,
                "y": (center[0] + offset_xy[1]) * self.MODEL_SCALE,
                "w": b[0],
                "h": b[1],
                "l": b[2],
                "score": float(score),
            }

            # discard negative values
            if bbox["w"] < 0:
                continue

            if bbox["h"] < 0:
                continue

            if bbox["l"] < 0:
                continue

            boxes.append(bbox)

        return boxes


class centernet(nn.Module):
    """
    Centernet simplified version
    Input = 1280x720 RGB image
    Backbone = ResNet-18 sampai layer4 (stride 32), jadi peta keluaran
    berukuran sekitar 23x40 (bukan 45x80). Cek dengan print(hm.shape).
    Output = heatmap utama [1,H,W], offset [2,H,W], w/h/l [3,H,W]
    """

    def __init__(self):
        super().__init__()

        # Resnet-18 as backbone.
        basemodel = torchvision.models.resnet18(weights=None)

        # Buang avgpool dan fc; sisanya sampai layer4 (512 channel, stride 32)
        self.base_model = nn.Sequential(*list(basemodel.children())[:-2])

        num_ch = 512
        head_conv = 64
        self.outc = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1, stride=1),
        )

        self.outo = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, stride=1),
        )

        self.outr = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 3, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = self.base_model(x)

        assert not torch.isnan(x).any()

        outc = self.outc(x)
        assert not torch.isnan(outc).any()

        outo = self.outo(x)
        assert not torch.isnan(outo).any()

        outr = self.outr(x)

        return outc, outo, outr
=== finetune_heatmap.py ===
"""
Fine-tune head heatmap (outc) CenterNet dengan label titik dari label_points.py.
Backbone dan head offset/ukuran dibekukan. Tidak ada augmentasi.
Pakai:
  python finetune_heatmap.py 13834567_1280x720.mp4 centernet-3d-bbox.pth --val-from 330
"""
import copy
import json
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from object_detector import centernet
from multi_object_tracking import K, to_world_from_camera
from utils import get_position_ground

STRIDE = 32
TOL_PX = 48  # deteksi dianggap benar bila dalam jarak ini dari titik label

ap = ArgumentParser()
ap.add_argument("video")
ap.add_argument("model")
ap.add_argument("--labels", default="labels.json")
ap.add_argument("--out", default="centernet-ft.pth")
ap.add_argument("--val-from", type=int, default=None, help="frame >= nilai ini dipakai validasi")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--sigma", type=float, default=0.8, help="lebar Gaussian target (satuan sel)")
args = ap.parse_args()

with open(args.labels, encoding="utf-8") as f:
    labels = {int(k): v["points"] for k, v in json.load(f).items() if v["done"]}
if not labels:
    raise SystemExit("Tidak ada frame berstatus SELESAI di " + args.labels)

model = centernet()
model.load_state_dict(torch.load(args.model, map_location="cpu"))
model.eval()
for p in model.parameters():
    p.requires_grad = False
for p in model.outc.parameters():
    p.requires_grad = True
base_head = copy.deepcopy(model.outc)

prep = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cap = cv2.VideoCapture(args.video)
feats = {}
print(f"Menghitung fitur {len(labels)} frame berlabel ...")
with torch.no_grad():
    for i in sorted(labels):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"Gagal membaca frame {i}")
        feats[i] = model.base_model(prep(frame).float().unsqueeze(0))
cap.release()
hh, ww = feats[next(iter(feats))].shape[-2:]
print(f"Ukuran heatmap: {hh} x {ww} (stride {STRIDE})")


def target_map(points):
    t = np.zeros((hh, ww), dtype=np.float32)
    ys, xs = np.mgrid[0:hh, 0:ww]
    for x, y in points:
        ix = min(max(int(x / STRIDE), 0), ww - 1)
        iy = min(max(int(y / STRIDE), 0), hh - 1)
        g = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * args.sigma ** 2))
        t = np.maximum(t, g.astype(np.float32))
        t[iy, ix] = 1.0
    return torch.from_numpy(t)


def focal_loss(logit, target):
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1.0 - pos
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * neg
    return -(pos_loss.sum() + neg_loss.sum()) / max(pos.sum().item(), 1.0)


def peaks(head, feat, conf):
    with torch.no_grad():
        hm = torch.sigmoid(head(feat))
        keep = F.max_pool2d(hm, 3, stride=1, padding=1) == hm
        hm = (hm * keep)[0, 0].numpy()
    out = []
    for cy, cx in np.argwhere(hm > conf):
        off = (0.5, 0.5)  # offset tidak dipakai di evaluasi; pakai tengah sel
        out.append(((cx + off[0]) * STRIDE, (cy + off[1]) * STRIDE, float(hm[cy, cx])))
    return out


def in_lane(x, y):
    g = get_position_ground(x, y, K, to_world_from_camera, 720)
    return abs(g[0]) <= 11


def evaluate(head, ids, conf):
    tp = fp = fn = 0
    for i in ids:
        gts = [tuple(p) for p in labels[i]]
        dets = [(x, y) for x, y, s in peaks(head, feats[i], conf) if in_lane(x, y)]
        used = set()
        for dx, dy in dets:
            best, bd = None, TOL_PX
            for k, (gx, gy) in enumerate(gts):
                d = np.hypot(dx - gx, dy - gy)
                if k not in used and d <= bd:
                    best, bd = k, d
            if best is None:
                fp += 1
            else:
                used.add(best)
                tp += 1
        fn += len(gts) - len(used)
    return tp, fp, fn


all_ids = sorted(labels)
if args.val_from is None:
    train_ids, val_ids = all_ids, []
else:
    train_ids = [i for i in all_ids if i < args.val_from]
    val_ids = [i for i in all_ids if i >= args.val_from]
print(f"Latih: {len(train_ids)} frame | validasi: {len(val_ids)} frame | "
      f"titik latih: {sum(len(labels[i]) for i in train_ids)}")

targets = {i: target_map(labels[i]) for i in all_ids}
opt = torch.optim.Adam(model.outc.parameters(), lr=args.lr)


def report(tag, head, ids):
    if not ids:
        return
    for conf in (0.2, 0.3, 0.5):
        tp, fp, fn = evaluate(head, ids, conf)
        print(f"  {tag:10} conf {conf}: benar {tp}, palsu {fp}, terlewat {fn}")


print("\nSebelum fine-tuning (model asli):")
report("latih", base_head, train_ids)
report("validasi", base_head, val_ids)

for ep in range(1, args.epochs + 1):
    order = np.random.permutation(train_ids)
    total = 0.0
    for i in order:
        loss = focal_loss(model.outc(feats[i])[0, 0], targets[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    if ep % 10 == 0 or ep == 1:
        print(f"epoch {ep:3}: loss rata-rata {total / len(train_ids):.4f}")

print("\nSetelah fine-tuning:")
report("latih", model.outc, train_ids)
report("validasi", model.outc, val_ids)

torch.save(model.state_dict(), args.out)
print(f"\nModel disimpan: {args.out}")

=== label_points.py ===
"""
Alat pelabelan titik mobil (satu klik per mobil).
Klik kiri  : tambah titik (titik kontak mobil dengan jalan)
Klik kanan : hapus titik terdekat
d / a      : frame +1 / -1        w / s : frame +5 / -5
u          : batalkan titik terakhir
SPASI      : tandai frame SELESAI (boleh 0 titik = contoh negatif)
m          : tampil/sembunyi titik deteksi model dari cache (kuning)
q / ESC    : simpan dan keluar
Penting: pada frame yang ditandai SELESAI, SEMUA mobil harus diberi titik,
karena mobil tanpa titik dianggap latar saat pelatihan.
"""
import json
import os
import pickle
from argparse import ArgumentParser

import cv2

ap = ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out", default="labels.json")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--cache", default=None, help="mis. det_low.pkl, untuk overlay titik model")
ap.add_argument("--scale", type=float, default=1.0, help="perkecil jendela jika layar kecil")
args = ap.parse_args()

data = {}
if os.path.exists(args.out):
    with open(args.out, encoding="utf-8") as f:
        data = json.load(f)

cache_frames = None
if args.cache:
    with open(args.cache, "rb") as f:  # hanya muat file yang Anda buat sendiri
        cache_frames = pickle.load(f)["frames"]

cap = cv2.VideoCapture(args.video)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
st = {"idx": min(max(args.start, 0), n - 1), "cache": True}


def rec(i):
    return data.setdefault(str(i), {"done": False, "points": []})


def save():
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f)


def on_mouse(event, x, y, flags, param):
    x, y = x / args.scale, y / args.scale
    r = rec(st["idx"])
    if event == cv2.EVENT_LBUTTONDOWN:
        r["points"].append([round(x, 1), round(y, 1)])
        save()
    elif event == cv2.EVENT_RBUTTONDOWN and r["points"]:
        j = min(range(len(r["points"])),
                key=lambda k: (r["points"][k][0] - x) ** 2 + (r["points"][k][1] - y) ** 2)
        r["points"].pop(j)
        save()


cv2.namedWindow("label")
cv2.setMouseCallback("label", on_mouse)

while True:
    i = st["idx"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok:
        break
    r = rec(i)
    if st["cache"] and cache_frames is not None:
        for b in cache_frames[i]:
            cv2.circle(frame, (int(b["x"]), int(b["y"])), 4, (0, 255, 255), -1)
            cv2.putText(frame, f"{b['score']:.2f}", (int(b["x"]) + 6, int(b["y"]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for k, (px, py) in enumerate(r["points"]):
        cv2.circle(frame, (int(px), int(py)), 7, (0, 0, 255), -1)
        cv2.putText(frame, str(k + 1), (int(px) + 9, int(py) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    n_done = sum(1 for v in data.values() if v["done"])
    status = "SELESAI" if r["done"] else "belum"
    cv2.putText(frame, f"frame {i}/{n-1} | titik {len(r['points'])} | {status} | frame selesai: {n_done}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if r["done"] else (255, 255, 255), 2)
    shown = frame if args.scale == 1.0 else cv2.resize(frame, None, fx=args.scale, fy=args.scale)
    cv2.imshow("label", shown)

    k = cv2.waitKey(30) & 0xFF
    if k in (ord("q"), 27):
        break
    elif k == ord("d"):
        st["idx"] = min(i + 1, n - 1)
    elif k == ord("a"):
        st["idx"] = max(i - 1, 0)
    elif k == ord("w"):
        st["idx"] = min(i + 5, n - 1)
    elif k == ord("s"):
        st["idx"] = max(i - 5, 0)
    elif k == ord("u") and r["points"]:
        r["points"].pop()
        save()
    elif k == 32:
        r["done"] = not r["done"]
        save()
    elif k == ord("m"):
        st["cache"] = not st["cache"]

save()
cap.release()
cv2.destroyAllWindows()
total = sum(len(v["points"]) for v in data.values() if v["done"])
print(f"Tersimpan di {args.out}: {sum(1 for v in data.values() if v['done'])} frame selesai, {total} titik")

=== lembar_hitungan.py ===
"""Lembar gambar: satu frame per hitungan, dengan penanda posisi saat melewati garis."""
import csv
from argparse import ArgumentParser

import cv2
import numpy as np

from multi_object_tracking import K, to_world_from_camera
from utils import project

ap = ArgumentParser()
ap.add_argument("video")
ap.add_argument("--log", default="counting_base03.csv")
ap.add_argument("--line-y", type=float, default=15.0)
ap.add_argument("--per", type=int, default=9)
args = ap.parse_args()

rows = list(csv.DictReader(open(args.log, encoding="utf-8")))
cap = cv2.VideoCapture(args.video)
to_cam = np.linalg.inv(to_world_from_camera)

tiles = []
for r in rows:
    i = int(r["frame"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    if not ok:
        continue
    h = fr.shape[0]
    pix = project(np.array([float(r["x_m"]), 0.0, args.line_y]), to_cam, K)
    cv2.circle(fr, (int(pix[0]), int(h - pix[1])), 14, (255, 0, 255), 4)
    cv2.putText(fr, f"ID {r['id']} | frame {i} | prediksi: {r['kelas']}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
    tiles.append(cv2.resize(fr, (640, 360)))

for s in range(0, len(tiles), args.per):
    chunk = tiles[s:s + args.per]
    while len(chunk) < args.per:
        chunk.append(np.zeros_like(tiles[0]))
    rows_img = [np.hstack(chunk[k:k + 3]) for k in range(0, args.per, 3)]
    name = f"lembar_{s // args.per + 1}.png"
    cv2.imwrite(name, np.vstack(rows_img))
    print("Disimpan:", name)

=== analisis_cache.py ===
"""
Lihat skor deteksi per frame dari cache (mis. det_low.pkl) pada area tertentu.
Tujuan: membandingkan skor mobil nyata yang hilang-timbul dengan skor deteksi palsu.

Pakai:
  python analisis_cache.py det_low.pkl
Ubah kotak piksel / rentang frame di bagian KONFIGURASI sesuai kebutuhan.
"""
import pickle
import sys

import numpy as np

from multi_object_tracking import K, to_world_from_camera
from utils import get_position_ground

# --- KONFIGURASI -------------------------------------------------------
# A. SUV lajur kanan (perkiraan dari tangkapan layar frame 300): kotak piksel
SUV_FRAMES = range(290, 331)
SUV_BOX = (700, 900, 420, 720)  # koordinat video

# B. Dua "truk" palsu dari log: track 3 (frame 43-72) dan track 7 (frame 119-162),
#    posisi dunia y ~ 23 m, x 1-3 m (dari tabel "tidak dihitung")
FALSE_FRAMES = list(range(230, 331))
FALSE_Y_RANGE = (8.0, 45.0)
FALSE_X_RANGE = (2.0, 5.5)
# -----------------------------------------------------------------------


def ground(box):
    p = get_position_ground(box["x"], box["y"], K, to_world_from_camera, 720)
    return float(p[0]), float(p[2])


def main(path):
    with open(path, "rb") as f:
        cache = pickle.load(f)  # hanya muat file yang Anda buat sendiri
    frames = cache["frames"]
    print(f"Cache: {path} | nms={cache['nms']} | skor minimum={cache['min_conf']} | {len(frames)} frame\n")

    print("A. SUV lajur kanan: deteksi di dalam kotak piksel (skor tertinggi per frame)")
    x0, x1, y0, y1 = SUV_BOX
    for i in SUV_FRAMES:
        hits = [b for b in frames[i] if x0 <= b["x"] <= x1 and y0 <= b["y"] <= y1]
        if hits:
            b = max(hits, key=lambda d: d["score"])
            print(f"  frame {i}: skor {b['score']:.3f}  piksel ({b['x']:.0f},{b['y']:.0f})")
        else:
            print(f"  frame {i}: tidak ada deteksi")

    print("\nB. Kandidat deteksi palsu (posisi dunia dalam rentang), skor tertinggi per frame")
    scores = []
    for i in FALSE_FRAMES:
        cands = []
        for b in frames[i]:
            x_m, y_m = ground(b)
            if FALSE_Y_RANGE[0] <= y_m <= FALSE_Y_RANGE[1] and FALSE_X_RANGE[0] <= x_m <= FALSE_X_RANGE[1]:
                cands.append((b, x_m, y_m))
        if cands:
            b, x_m, y_m = max(cands, key=lambda c: c[0]["score"])
            scores.append(b["score"])
            print(f"  frame {i}: skor {b['score']:.3f}  piksel ({b['x']:.0f},{b['y']:.0f})  dunia x={x_m:.1f} y={y_m:.1f}")
    if scores:
        print(f"\n  Ringkasan B: n={len(scores)}, skor min/median/maks = "
              f"{min(scores):.3f} / {np.median(scores):.3f} / {max(scores):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "det_low.pkl")

=== bandingkan_ukuran.py ===
"""Bandingkan panjang terukur deteksi fine-tune dengan deteksi model asli di lokasi yang sama."""
import pickle

import numpy as np

from multi_object_tracking import K, to_world_from_camera
from utils import get_position_ground

STRIDE, TOL_PX, MIN_SCORE = 32, 48.0, 0.3


def load(p):
    with open(p, "rb") as f:  # hanya muat file yang Anda buat sendiri
        return pickle.load(f)["frames"]


base, ft = load("det_low.pkl"), load("det_ft.pkl")
cam = to_world_from_camera[:3, 3]


def meters(b):
    g = get_position_ground(b["x"], b["y"], K, to_world_from_camera, 720)
    return g, float(np.linalg.norm(g - cam))


same, diff, new = [], [], []
for fb, ff in zip(base, ft):
    for b in ff:
        if b["score"] < MIN_SCORE:
            continue
        g, d = meters(b)
        if abs(g[0]) > 11:
            continue
        l_ft = b["l"] * d
        best, bd = None, TOL_PX
        for c in fb:
            dist = np.hypot(b["x"] - c["x"], b["y"] - c["y"])
            if dist <= bd:
                best, bd = c, dist
        if best is None:
            new.append(l_ft)
            continue
        l_base = best["l"] * meters(best)[1]
        same_cell = (int(b["x"] // STRIDE), int(b["y"] // STRIDE)) == (int(best["x"] // STRIDE), int(best["y"] // STRIDE))
        (same if same_cell else diff).append((l_ft, l_base))


def report(name, pairs):
    if not pairs:
        print(f"{name}: tidak ada")
        return
    a = np.array(pairs)
    print(f"{name}: n={len(a)} | panjang fine-tune median {np.median(a[:, 0]):.2f} m | "
          f"model asli median {np.median(a[:, 1]):.2f} m | rasio median {np.median(a[:, 0] / a[:, 1]):.2f}")


print("Deteksi fine-tune (skor >= 0.3, dalam jalur) dibandingkan dengan deteksi model asli terdekat:")
report("Sel sama   ", same)
report("Sel berbeda", diff)
if new:
    print(f"Tanpa pasangan di model asli: n={len(new)} | panjang median {np.median(new):.2f} m")

=== cek_fp.py ===
"""
Evaluasi titik dengan offset (sama seperti pred2box) dan simpan frame dengan deteksi palsu terbanyak.
Pakai: python cek_fp.py VIDEO MODEL --conf 0.3 --from 330 --to 472 --out cek_base
"""
import json
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from object_detector import centernet
from multi_object_tracking import K, to_world_from_camera
from utils import get_position_ground

STRIDE = 32
TOL_PX = 48

ap = ArgumentParser()
ap.add_argument("video")
ap.add_argument("model")
ap.add_argument("--labels", default="labels.json")
ap.add_argument("--conf", type=float, default=0.3)
ap.add_argument("--from", dest="f0", type=int, default=330)
ap.add_argument("--to", dest="f1", type=int, default=472)
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--out", default="cek_fp")
args = ap.parse_args()

with open(args.labels, encoding="utf-8") as f:
    labels = {int(k): v["points"] for k, v in json.load(f).items() if v["done"]}

model = centernet()
model.load_state_dict(torch.load(args.model, map_location="cpu"))
model.eval()
prep = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def in_lane(x, y):
    g = get_position_ground(x, y, K, to_world_from_camera, 720)
    return abs(g[0]) <= 11


os.makedirs(args.out, exist_ok=True)
cap = cv2.VideoCapture(args.video)
tp = fp = fn = 0
per_frame = []
for i in range(args.f0, args.f1 + 1):
    if i not in labels:
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok:
        break
    with torch.no_grad():
        hm, off, _ = model(prep(frame).float().unsqueeze(0))
        hm = torch.sigmoid(hm)
        keep = F.max_pool2d(hm, 3, stride=1, padding=1) == hm
        hm = (hm * keep)[0, 0].numpy()
        off = off[0].numpy()
    dets = []
    for cy, cx in np.argwhere(hm > args.conf):
        x = (cx + off[0, cy, cx]) * STRIDE
        y = (cy + off[1, cy, cx]) * STRIDE
        if in_lane(x, y):
            dets.append((x, y, float(hm[cy, cx])))
    gts = labels[i]
    used, matched, fps = set(), [], []
    for x, y, s in dets:
        best, bd = None, TOL_PX
        for k, (gx, gy) in enumerate(gts):
            d = np.hypot(x - gx, y - gy)
            if k not in used and d <= bd:
                best, bd = k, d
        if best is None:
            fps.append((x, y, s))
        else:
            used.add(best)
            matched.append((x, y, s))
    tp += len(matched)
    fp += len(fps)
    fn += len(gts) - len(used)
    per_frame.append((len(fps), i, gts, matched, fps, used))

print(f"{args.model} | conf {args.conf} | frame {args.f0}-{args.f1}: benar {tp}, palsu {fp}, terlewat {fn}")
per_frame.sort(key=lambda t: -t[0])
for n_fp, i, gts, matched, fps, used in per_frame[: args.top]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    _, img = cap.read()
    for k, (gx, gy) in enumerate(gts):
        cv2.circle(img, (int(gx), int(gy)), 8, (0, 0, 255), -1)
        if k not in used:
            cv2.circle(img, (int(gx), int(gy)), 22, (0, 0, 255), 3)
    for x, y, s in matched:
        cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0), -1)
    for x, y, s in fps:
        cv2.circle(img, (int(x), int(y)), 6, (0, 255, 255), -1)
        cv2.putText(img, f"{s:.2f}", (int(x) + 8, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite(f"{args.out}/frame_{i:04d}_palsu{n_fp}.png", img)
print(f"Gambar disimpan di {args.out}/")

=== calibrate_tuner.py ===
#!/usr/bin/env python3
"""
Alat penyetel kalibrasi kamera dengan slider.

Menampilkan di atas satu frame video:
  - grid jalan (garis searah jalan + garis melintang) yang bisa digeser
  - garis horizon (dari pitch dan focal saat ini)
  - satu kotak mobil ukuran standar (1.85 x 1.5 x 4.5 m)

Konvensi koordinat sama dengan utils.py / multi_object_tracking.py:
  dunia  : x = kanan, y = atas, z = maju
  kamera : x = kanan, y = atas, z = maju
  pitch  : make_euler(-pitch_deg * pi/180, 0)  (pitch positif = kamera menunduk)

Pakai:
    python calibrate_tuner.py video.mp4 --frame 100
Tekan  q  untuk selesai dan mencetak nilai siap salin. Tekan  ESC  untuk batal.
"""

from argparse import ArgumentParser

import cv2
import numpy as np

WIN = "Calibrate tuner"

CAR_W, CAR_H, CAR_L = 1.85, 1.5, 4.5  # mobil standar (m)


def rot_pitch(pitch_deg):
    """Sama dengan make_euler(-pitch, 0) di utils.py."""
    a = -pitch_deg * np.pi / 180.0
    return np.array(
        [[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]], dtype=float
    )


class Cam:
    def __init__(self, pitch_deg, height, focal, w, h):
        self.R = rot_pitch(pitch_deg)  # camera -> world
        self.t = np.array([0.0, height, 0.0])
        self.f = focal
        self.cx, self.cy = w / 2.0, h / 2.0
        self.w, self.h = w, h
        self.pitch = pitch_deg

    def project(self, pw):
        """Titik dunia -> piksel (u, v). None jika di belakang kamera."""
        pc = self.R.T.dot(np.asarray(pw, dtype=float) - self.t)
        if pc[2] < 0.1:
            return None
        u = self.f * pc[0] / pc[2] + self.cx
        v_up = self.f * pc[1] / pc[2] + self.cy
        return (int(round(u)), int(round(self.h - v_up)))

    def horizon_row(self):
        return int(round(self.cy - self.f * np.tan(self.pitch * np.pi / 180.0)))


def draw_polyline(img, cam, pts, color, thickness=1):
    prev = None
    for p in pts:
        q = cam.project(p)
        if q is not None and prev is not None:
            cv2.line(img, prev, q, color, thickness, cv2.LINE_AA)
        prev = q


def draw_grid(img, cam, off_x, off_z, lane_w, z_max=150):
    color = (255, 200, 100)  # biru muda (BGR)
    zs = np.arange(2.0, z_max, 1.0)

    # garis searah jalan (x konstan)
    for k in range(-5, 6):
        x = off_x + k * lane_w
        draw_polyline(img, cam, [(x, 0, z) for z in zs], color, 1)

    # garis melintang tiap 10 m
    x_lo, x_hi = off_x - 5 * lane_w, off_x + 5 * lane_w
    xs = np.linspace(x_lo, x_hi, 12)
    z = off_z if off_z > 2 else off_z + 10
    while z < z_max:
        draw_polyline(img, cam, [(x, 0, z) for x in xs], (255, 160, 60), 1)
        z += 10.0


def draw_horizon(img, cam):
    row = cam.horizon_row()
    if 0 <= row < img.shape[0]:
        for x in range(0, img.shape[1], 24):
            cv2.line(img, (x, row), (x + 12, row), (0, 255, 255), 1)
        cv2.putText(img, "horizon", (8, row - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)


def draw_box(img, cam, x, z):
    hw, hl = CAR_W / 2, CAR_L / 2
    bot = [(x - hw, 0, z - hl), (x + hw, 0, z - hl), (x + hw, 0, z + hl), (x - hw, 0, z + hl)]
    top = [(px, CAR_H, pz) for px, _, pz in bot]
    pb = [cam.project(p) for p in bot]
    pt = [cam.project(p) for p in top]
    edges = []
    for i in range(4):
        j = (i + 1) % 4
        edges += [(pb[i], pb[j]), (pt[i], pt[j]), (pb[i], pt[i])]
    for a, b in edges:
        if a is not None and b is not None:
            cv2.line(img, a, b, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    ap = ArgumentParser(description="Penyetel kalibrasi kamera")
    ap.add_argument("video", type=str)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--pitch", type=float, default=10.0, help="pitch awal (derajat, menunduk)")
    ap.add_argument("--height", type=float, default=8.5, help="tinggi kamera awal (m)")
    ap.add_argument("--focal", type=float, default=770.19, help="focal awal (px)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Gagal membaca frame dari video")
    H, W = frame.shape[:2]

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    nop = lambda v: None
    #            nama        default                      maksimum
    sliders = [
        ("PITCH x10",    int(round(args.pitch * 10)),   600),   # 0..60 derajat
        ("HEIGHT x10",   int(round(args.height * 10)),  400),   # 0..40 m
        ("FOCAL",        int(round(args.focal)),        3000),  # px
        ("REF_X x10",    200,                           400),   # -20..+20 m
        ("REF_Z",        25,                            150),   # m
        ("GRID_OX x10",  50,                            100),   # -5..+5 m (geser grid ke samping)
        ("GRID_OZ x10",  0,                             100),   # 0..10 m (geser grid ke depan)
        ("LANE_W x10",   35,                            60),    # lebar lajur 1..6 m
    ]
    for name, d, m in sliders:
        cv2.createTrackbar(name, WIN, d, m, nop)

    def val(n):
        return cv2.getTrackbarPos(n, WIN)

    result = None
    while True:
        pitch = val("PITCH x10") / 10.0
        height = max(val("HEIGHT x10"), 10) / 10.0
        focal = max(val("FOCAL"), 100)
        ref_x = (val("REF_X x10") - 200) / 10.0
        ref_z = max(val("REF_Z"), 3)
        off_x = (val("GRID_OX x10") - 50) / 10.0
        off_z = val("GRID_OZ x10") / 10.0
        lane_w = max(val("LANE_W x10"), 10) / 10.0

        cam = Cam(pitch, height, focal, W, H)
        canvas = frame.copy()
        draw_grid(canvas, cam, off_x, off_z, lane_w)
        draw_horizon(canvas, cam)
        draw_box(canvas, cam, ref_x, ref_z)

        # Panel label: urutan sama persis dengan urutan slider dari atas ke bawah
        rows = [
            f"1 PITCH    {pitch:5.1f} deg",
            f"2 HEIGHT   {height:5.1f} m",
            f"3 FOCAL    {focal:5d} px",
            f"4 REF_X    {ref_x:+5.1f} m",
            f"5 REF_Z    {ref_z:5d} m",
            f"6 GRID_OX  {off_x:+5.1f} m",
            f"7 GRID_OZ  {off_z:5.1f} m",
            f"8 LANE_W   {lane_w:5.1f} m",
        ]
        panel_w, row_h = 250, 26
        panel_h = row_h * len(rows) + 12
        overlay = canvas.copy()
        cv2.rectangle(overlay, (6, 6), (6 + panel_w, 6 + panel_h), (0, 0, 0), -1)
        canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)
        for i, text in enumerate(rows):
            cv2.putText(canvas, text, (14, 30 + i * row_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WIN, canvas)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            result = (pitch, height, focal)
            break
        if k == 27 or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    if result is None:
        print("Dibatalkan.")
        return

    pitch, height, focal = result
    print("\n# Salin ke multi_object_tracking.py:")
    print(f"pitch_matrix = make_euler(-{pitch:.2f} * np.pi / 180, 0)")
    print(f"to_world_from_camera[:3, 3] = np.array([0, {height:.2f}, 0])")
    print(f"K = np.array([[{focal:.2f}, 0.0, {W / 2:.1f}], "
          f"[0.0, {focal:.2f}, {H / 2:.1f}], [0.0, 0.0, 1.0]])")


if __name__ == "__main__":
    main()