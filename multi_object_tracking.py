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