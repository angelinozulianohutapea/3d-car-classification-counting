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