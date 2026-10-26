"""Menggambar hasil: kotak 3D, label, garis counting, panel hitungan, tanda debug."""

import cv2
import numpy as np

import config as cfg
from box3d import box_corners
from geometry import project, world_to_image

# Warna BGR untuk membedakan ID
COLORS = [
    (0, 0, 128), (0, 0, 255), (0, 128, 0), (0, 128, 128), (0, 128, 255),
    (0, 255, 0), (0, 255, 128), (0, 255, 255), (128, 0, 0), (128, 0, 128),
    (128, 0, 255), (128, 128, 0), (128, 128, 128), (128, 128, 255), (128, 255, 0),
    (128, 255, 128), (128, 255, 255), (255, 0, 0), (255, 0, 128), (255, 0, 255),
    (255, 128, 0), (255, 128, 128), (255, 128, 255), (255, 255, 0), (255, 255, 128),
    (255, 255, 255),
]


def draw_label(frame, lines, x, y, color, scale=0.6, thickness=1):
    """Label beberapa baris berlatar warna; bagian bawah di y, dipusatkan di x, tetap dalam frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 4
    sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
    w = max(s[0] for s in sizes) + 2 * pad
    line_h = max(s[1] for s in sizes) + 6
    h = line_h * len(lines) + pad
    x0 = int(max(0, min(x - w / 2, frame.shape[1] - w)))
    y0 = int(max(0, min(y - h, frame.shape[0] - h)))
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), color, thickness=-1)

    b, g, r = color
    text_color = (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 140 else (255, 255, 255)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (x0 + pad, y0 + (i + 1) * line_h - 2), font, scale,
                    text_color, thickness, cv2.LINE_AA)


def draw_oriented_box(frame, track, color, fill=True):
    """Kotak 3D berorientasi milik track."""
    h_img = frame.shape[0]
    corners = box_corners(track.x(), track.y(), track.dims(), track.yaw)
    pts = [world_to_image(c, h_img) for c in corners]

    if fill:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(pts[0:4], dtype=np.int32)], color)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        cv2.line(frame, pts[a], pts[b], color, 2)
    return frame


def draw_tracks(frame, tracker):
    """Kotak 3D dan label (ID, kelas) untuk semua track yang layak tampil."""
    for t in tracker.visible_tracks():
        color = COLORS[t.display_id % len(COLORS)]
        frame = draw_oriented_box(frame, t, color)

        # label di atas titik tengah atap kotak; objek jauh = font lebih kecil
        top = project(np.array([t.x(), t.dims()[1], t.y()]))
        tx, ty = top[0], frame.shape[0] - top[1]
        scale = float(np.clip(0.8 - t.y() / 80.0, 0.45, 0.8))
        draw_label(frame, [f"ID {t.display_id} ({t.cls})"], tx, ty - 6, color, scale=scale)
    return frame


def draw_counting_line(frame, line_y, line_y_away=None):
    """Garis virtual: merah = jalur mendekat (kanan median), oranye = jalur menjauh (kiri)."""
    h = frame.shape[0]
    if line_y_away is None:
        line_y_away = line_y
    segments = [
        (line_y, cfg.MEDIAN_X, cfg.LANE_X_MAX, (0, 0, 255)),
    ]
    for z, x0, x1, color in segments:
        p0 = world_to_image(np.array([x0, 0.0, z]), h)
        p1 = world_to_image(np.array([x1, 0.0, z]), h)
        cv2.line(frame, p0, p1, color, 3)
    return frame


def draw_counts(frame, counts, dir_counts=None):
    """Panel hitungan per kelas (dan per arah bila diberikan) di pojok kiri bawah."""
    lines = [f"Total: {sum(counts.values())}"] + [f"{c}: {counts[c]}" for c in cfg.CLASSES]
    if dir_counts:
        lines += [f"{d}: {sum(v.values())}" for d, v in dir_counts.items()]
    h = frame.shape[0]
    box_h = 26 * len(lines) + 10
    cv2.rectangle(frame, (10, h - box_h - 10), (230 if dir_counts else 190, h - 10), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (18, h - box_h + 18 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


def draw_debug(frame, marks):
    """Deteksi mentah: hijau = lolos, kuning = lolos skor rendah, merah = ditolak (+alasan)."""
    for x, y, reason, low in marks:
        pt = (int(x), int(y))
        if reason is None:
            cv2.circle(frame, pt, 6, (0, 255, 255) if low else (0, 255, 0), -1)
        else:
            cv2.circle(frame, pt, 6, (0, 0, 255), -1)
            cv2.putText(frame, reason, (pt[0] + 8, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame


def draw_count_marks(frame, tracker, counter, frame_idx, hold=30):
    """Penanda 'HITUNG' di atas kendaraan selama `hold` frame sejak terhitung.
    Oranye = menjauh, hijau = mendekat."""
    for t in tracker.tracks:
        rec = counter.counted_at.get(t.id)
        if rec is None or t.display_id is None:
            continue
        f0, direction = rec
        if not (0 <= frame_idx - f0 < hold):
            continue
        color = (0, 165, 255) if direction == "menjauh" else (0, 200, 0)
        top = project(np.array([t.x(), t.dims()[1], t.y()]))
        tx, ty = top[0], frame.shape[0] - top[1]
        draw_label(frame, [f"HITUNG {direction}"], tx, ty - 34, color, scale=0.55)
    return frame


def draw_frame_id(frame, frame_id):
    cv2.putText(frame, f"Frame {frame_id}", (frame.shape[1] - 150, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
