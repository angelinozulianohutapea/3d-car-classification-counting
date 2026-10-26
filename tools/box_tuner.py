#!/usr/bin/env python3
"""
Penyetel kotak 3D pada SATU frame (cepat, tanpa memproses ulang video).

Tujuan: mencari kalibrasi kamera (PITCH, HEIGHT, FOCAL) dan ukuran mobil yang
membuat kotak 3D pas dengan mobil di gambar.

Cara pakai:
  1. KLIK kiri pada titik kontak ban mobil dengan jalan (tengah bawah mobil).
     Kotak hijau muncul di posisi itu (dihitung dengan proyeksi balik yang
     SAMA dengan geometry.py).
  2. Geser slider PITCH / HEIGHT / FOCAL sampai garis grid sejajar marka jalan
     dan kotak pas. Lalu slider W/H/L/YAW untuk ukuran dan arah.
  3. Klik mobil lain (dekat dan jauh) untuk memeriksa: kalibrasi yang benar
     membuat kotak pas di SEMUA jarak, bukan hanya di satu mobil.
  4. q = selesai dan cetak nilai siap salin ke config.py. ESC = batal.

Opsional --model: titik deteksi model digambar (kuning) beserta kotak ukuran
terukur model (cyan, yaw 0) pada kalibrasi saat ini.

Tombol: g grid on/off | m deteksi on/off | p cetak tabel deteksi |
        i/k geser mobil maju/mundur 0.25 m | j/l geser kiri/kanan 0.1 m

Pakai (dari folder proyek):
  python -m tools.box_tuner data/video.mp4 --frame 285
  python -m tools.box_tuner data/video.mp4 --frame 285 --model centernet-3d-bbox.pth
  python -m tools.box_tuner experiments/out/frame_286.png
"""

import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg  # noqa: E402
from box3d import box_corners  # noqa: E402

WIN = "Box tuner"
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


class Cam:
    """Kamera yang identik dengan geometry.py, tetapi parameternya bisa diubah."""

    def __init__(self, pitch_deg, height, focal, w, h):
        a = -pitch_deg * np.pi / 180.0
        self.R = np.array([[1, 0, 0],
                           [0, np.cos(a), np.sin(a)],
                           [0, -np.sin(a), np.cos(a)]], dtype=float)
        self.t = np.array([0.0, height, 0.0])
        self.f, self.w, self.h = focal, w, h
        self.cx, self.cy = w / 2.0, h / 2.0
        self.pitch = pitch_deg

    def project(self, pw):
        """Titik dunia -> piksel OpenCV (u, v dari atas). None bila di belakang kamera."""
        pc = self.R.T.dot(np.asarray(pw, dtype=float) - self.t)
        if pc[2] < 0.1:
            return None
        u = self.f * pc[0] / pc[2] + self.cx
        v_up = self.f * pc[1] / pc[2] + self.cy
        return (int(round(u)), int(round(self.h - v_up)))

    def ground(self, x, y):
        """Piksel OpenCV -> posisi di bidang jalan (x, 0, z); sama dengan get_position_ground."""
        d = self.R.dot(np.array([(x - self.cx) / self.f, (self.h - y - self.cy) / self.f, 1.0]))
        if d[1] >= -1e-6:
            return None  # sinar tidak menyentuh jalan (di atas horizon)
        lam = -self.t[1] / d[1]
        return self.t + lam * d

    def horizon_row(self):
        return int(round(self.cy - self.f * np.tan(self.pitch * np.pi / 180.0)))


def draw_polyline(img, cam, pts, color, thickness=1):
    prev = None
    for p in pts:
        q = cam.project(p)
        if q is not None and prev is not None:
            cv2.line(img, prev, q, color, thickness, cv2.LINE_AA)
        prev = q


def draw_grid(img, cam, z_max=120):
    zs = np.arange(2.0, z_max, 1.0)
    for x in range(-12, 13, 2):  # garis searah jalan tiap 2 m
        draw_polyline(img, cam, [(x, 0, z) for z in zs], (255, 200, 100), 1)
    for z in range(5, int(z_max), 5):  # garis melintang tiap 5 m
        draw_polyline(img, cam, [(x, 0, z) for x in (-12, 12)], (255, 160, 60), 1)
        q = cam.project((-12, 0, z))
        if q is not None and 0 <= q[0] < img.shape[1] and 0 <= q[1] < img.shape[0] and z % 10 == 0:
            cv2.putText(img, f"{z}m", (q[0] + 4, q[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 200, 100), 1, cv2.LINE_AA)


def draw_horizon(img, cam):
    row = cam.horizon_row()
    if 0 <= row < img.shape[0]:
        for x in range(0, img.shape[1], 24):
            cv2.line(img, (x, row), (x + 12, row), (0, 255, 255), 1)


def draw_car(img, cam, cx, cz, dims, yaw, color, thickness=2, fill=False):
    pts = [cam.project(c) for c in box_corners(cx, cz, dims, yaw)]
    if any(p is None for p in pts):
        return
    if fill:
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.array(pts[0:4], dtype=np.int32)], color)
        img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    for a, b in EDGES:
        cv2.line(img, pts[a], pts[b], color, thickness, cv2.LINE_AA)


def detections_in_world(cam, boxes):
    """Deteksi model (rasio) -> daftar (x_px, y_px, x_m, z_m, w_m, h_m, l_m, skor)."""
    out = []
    for b in boxes:
        g = cam.ground(b["x"], b["y"])
        if g is None:
            continue
        depth = float(np.linalg.norm(g - cam.t))
        out.append((b["x"], b["y"], float(g[0]), float(g[2]),
                    b["w"] * depth, b["h"] * depth, b["l"] * depth, b["score"]))
    return out


def render(frame, cam, car, params, show_grid=True, dets=None):
    """Gambar satu tampilan. car = (x, z) atau None; params = dict W,H,L,yaw(rad)."""
    canvas = frame.copy()
    if show_grid:
        draw_grid(canvas, cam)
    draw_horizon(canvas, cam)

    if dets:
        for xp, yp, xm, zm, wm, hm, lm, s in dets:
            cv2.circle(canvas, (int(xp), int(yp)), 5, (0, 255, 255), -1)
            draw_car(canvas, cam, xm, zm, (min(wm, 3.0), hm, lm), 0.0, (255, 255, 0), 1)

    if car is not None:
        dims = (params["W"], params["H"], params["L"])
        draw_car(canvas, cam, car[0], car[1], dims, params["yaw"], (0, 255, 0), 2, fill=True)
        q = cam.project((car[0], 0.0, car[1]))
        if q is not None:
            cv2.circle(canvas, q, 4, (0, 0, 255), -1)

    rows = [f"PITCH  {cam.pitch:5.1f} deg",
            f"HEIGHT {cam.t[1]:5.1f} m",
            f"FOCAL  {cam.f:5.0f} px"]
    if car is not None:
        rows += [f"mobil x={car[0]:+5.1f} z={car[1]:5.1f} m",
                 f"W {params['W']:.2f} H {params['H']:.2f} L {params['L']:.2f} m",
                 f"yaw {np.degrees(params['yaw']):+4.0f} deg"]
    else:
        rows.append("klik titik ban mobil di jalan")
    overlay = canvas.copy()
    cv2.rectangle(overlay, (6, 6), (290, 14 + 24 * len(rows)), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)
    for i, t in enumerate(rows):
        cv2.putText(canvas, t, (14, 28 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def load_frame(src, idx):
    if src.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        frame = cv2.imread(src)
    else:
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        frame = frame if ok else None
    if frame is None:
        raise SystemExit(f"Gagal membaca frame dari {src}")
    return cv2.resize(frame, (cfg.IMG_W, cfg.IMG_H))


def main():
    ap = ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("source", help="video atau gambar (png/jpg)")
    ap.add_argument("--frame", type=int, default=285, help="indeks frame OpenCV (dari 0)")
    ap.add_argument("--model", default=None, help="bobot CenterNet; tampilkan deteksi model")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    frame = load_frame(args.source, args.frame)
    H, W = frame.shape[:2]

    boxes = []
    if args.model:
        from detector import Detector
        boxes = Detector(args.model, args.conf).detect(frame)
        print(f"{len(boxes)} deteksi pada frame ini")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    t = cfg.TEMPLATES["car"]
    sliders = [
        ("PITCH x10", int(round(cfg.PITCH_DEG * 10)), 600),
        ("HEIGHT x10", int(round(cfg.CAM_HEIGHT_M * 10)), 400),
        ("FOCAL", int(round(cfg.FOCAL_PX)), 3000),
        ("W x100", int(round(t["w"] * 100)), 350),
        ("H x100", int(round(t["h"] * 100)), 500),
        ("L x100", int(round(t["l"] * 100)), 1500),
        ("YAW deg+40", 40, 80),
    ]
    for name, d, m in sliders:
        cv2.createTrackbar(name, WIN, d, m, lambda v: None)

    def val(n):
        return cv2.getTrackbarPos(n, WIN)

    state = {"car": None, "grid": True, "dets": bool(boxes)}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cam = Cam(val("PITCH x10") / 10.0, max(val("HEIGHT x10"), 10) / 10.0,
                      max(val("FOCAL"), 100), W, H)
            g = cam.ground(x, y)
            if g is not None:
                state["car"] = [float(g[0]), float(g[2])]
                print(f"Mobil di x={g[0]:+.1f} z={g[2]:.1f} m (piksel {x},{y})")

    cv2.setMouseCallback(WIN, on_mouse)

    result = None
    while True:
        cam = Cam(val("PITCH x10") / 10.0, max(val("HEIGHT x10"), 10) / 10.0,
                  max(val("FOCAL"), 100), W, H)
        params = {"W": max(val("W x100"), 100) / 100.0, "H": max(val("H x100"), 100) / 100.0,
                  "L": max(val("L x100"), 200) / 100.0,
                  "yaw": np.radians(val("YAW deg+40") - 40)}
        dets = detections_in_world(cam, boxes) if (boxes and state["dets"]) else None
        canvas = render(frame, cam, state["car"], params, state["grid"], dets)
        cv2.imshow(WIN, canvas)

        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            result = (cam.pitch, cam.t[1], cam.f, params)
            break
        if k == 27 or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == ord("g"):
            state["grid"] = not state["grid"]
        elif k == ord("m"):
            state["dets"] = not state["dets"]
        elif k == ord("p") and boxes:
            print("\n  x_px  y_px |   x_m    z_m |  w_m  h_m  l_m | skor")
            for xp, yp, xm, zm, wm, hm, lm, s in sorted(detections_in_world(cam, boxes), key=lambda r: r[3]):
                print(f"  {xp:4.0f}  {yp:4.0f} | {xm:+6.1f} {zm:6.1f} | {wm:4.2f} {hm:4.2f} {lm:4.2f} | {s:.2f}")
        elif state["car"] is not None and k in (ord("i"), ord("k"), ord("j"), ord("l")):
            state["car"][1] += 0.25 if k == ord("i") else -0.25 if k == ord("k") else 0.0
            state["car"][0] += 0.1 if k == ord("l") else -0.1 if k == ord("j") else 0.0

    cv2.destroyAllWindows()
    if result is None:
        print("Dibatalkan.")
        return

    pitch, height, focal, p = result
    print("\n# Salin ke config.py (kalibrasi kamera):")
    print(f"PITCH_DEG = {pitch:.2f}")
    print(f"CAM_HEIGHT_M = {height:.2f}")
    print(f"FOCAL_PX = {focal:.2f}")
    print("\n# Ukuran mobil yang pas di frame ini (rujukan untuk TEMPLATES['car'], bila perlu):")
    print(f'    "car": {{"w": {p["W"]:.2f}, "h": {p["H"]:.2f}, "l": {p["L"]:.2f}}},')


if __name__ == "__main__":
    main()
