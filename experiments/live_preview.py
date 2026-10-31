"""
Preview langsung deteksi + 3D bounding box untuk menyetel parameter.

Tombol:
  spasi  pause/lanjut        a / d  mundur/maju 1 frame (saat pause)
  + / -  ambang conf         s      ganti kombinasi skala
  g      gamma on/off        f      flip on/off
  b      3D box on/off       p      titik deteksi mentah on/off
  c      simpan frame mentah ke experiments/out/frame_NNN.png (NNN = nomor di layar)
  q      keluar
Nomor frame di layar dihitung dari 1 (OpenCV dari 0, jadi frame NNN = indeks NNN-1).

Jalankan dari folder proyek:
  python experiments/live_preview.py --video 13834567_1280x720.mp4 --model centernet-3d-bbox.pth
"""
import argparse, copy, os, sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as cfg
from detector import Detector
from detection_filter import to_world_detections
from kalman_filter import TIME_INTERVAL_FALLBACK
from tracker import Tracker
from visualize import draw_tracks

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="data/video.mp4")
ap.add_argument("--model", default="centernet-3d-bbox.pth")
ap.add_argument("--every", type=int, default=3, help="saat jalan, deteksi tiap N frame")
ap.add_argument("--no-box3d", action="store_true",
                help="mulai tanpa 3D box (tekan b untuk menyalakan)")
a = ap.parse_args()

out_dir = os.path.join(ROOT, "experiments", "out")
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(a.video)
if not cap.isOpened():
    raise SystemExit(f"Video tidak terbuka: {a.video}")

fps = cap.get(cv2.CAP_PROP_FPS)
# tracker diperbarui tiap N frame (saat jalan), jadi selang waktunya N / fps
dt = (max(1, a.every) / fps) if fps and fps > 1 else TIME_INTERVAL_FALLBACK

det = Detector(a.model)
gamma_on = det.gamma
SCALE_SETS = [(1.0,), (1.0, 1.5), (1.0, 1.5, 2.0)]
si = SCALE_SETS.index(det.scales) if det.scales in SCALE_SETS else 1
det.scales = SCALE_SETS[si]


def new_tracker():
    return Tracker(dt)


def track_frame(tracker, snap, tracked_n, n, boxes, conf):
    """Satu langkah tracking. Bila frame yang sama diproses ulang (mis. ganti conf saat
    pause), tracker dikembalikan dulu ke kondisi sebelum frame itu."""
    if n == tracked_n and snap is not None:
        tracker = copy.deepcopy(snap)
    else:
        snap = copy.deepcopy(tracker)
    good = [b for b in boxes if b["score"] >= conf]
    dets, _marks, _rejected = to_world_detections(good, None)
    tracker.update(dets, conf_high=None)
    tracker.assign_display_ids()
    return tracker, snap, n


box3d = not a.no_box3d
show_pts = not box3d
tracker, snap, tracked_n = new_tracker(), None, -1

paused, frame, n, boxes, dirty = False, None, 0, [], True
while True:
    if not paused or frame is None:
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            n = 0
            tracker, snap, tracked_n = new_tracker(), None, -1
            continue
        frame, n, dirty = cv2.resize(f, (1280, 720)), n + 1, True

    if dirty and (paused or n % a.every == 0):
        if paused:
            boxes = det.detect(frame)          # setelan penuh (skala/flip sesuai tombol)
        else:
            sv = (det.scales, det.flip)        # mode cepat saat video jalan
            det.scales, det.flip = (1.0,), False
            boxes = det.detect(frame)
            det.scales, det.flip = sv
        dirty = False
        if box3d:
            tracker, snap, tracked_n = track_frame(tracker, snap, tracked_n, n, boxes, det.conf)

    vis = frame.copy()
    if box3d:
        vis = draw_tracks(vis, tracker)
    if show_pts:
        for b in boxes:
            c = (0, 255, 0) if b["score"] >= 0.3 else (0, 200, 255)
            cv2.circle(vis, (int(b["x"]), int(b["y"])), 7, c, -1)
            cv2.putText(vis, f'{b["score"]:.2f}', (int(b["x"]) + 8, int(b["y"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    info = (f"{'PENUH' if paused else 'CEPAT'} FRAME={n} conf={det.conf:.2f} skala={det.scales} "
            f"gamma={det.gamma} flip={det.flip} n={len(boxes)} 3D={'on' if box3d else 'off'}")
    cv2.rectangle(vis, (0, 0), (1280, 30), (0, 0, 0), -1)
    cv2.putText(vis, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.imshow("preview", vis)

    k = cv2.waitKey(1 if not paused else 30) & 0xFF
    if k == ord("q"):
        break
    elif k == ord(" "):
        paused = not paused
    elif k in (ord("+"), ord("=")):
        det.conf, dirty = min(0.95, det.conf + 0.05), True
    elif k in (ord("-"), ord("_")):
        det.conf, dirty = max(0.05, det.conf - 0.05), True
    elif k == ord("s"):
        si = (si + 1) % len(SCALE_SETS)
        det.scales, dirty = SCALE_SETS[si], True
    elif k == ord("g"):
        det.gamma = 1.0 if det.gamma != 1.0 else (gamma_on if gamma_on != 1.0 else 2.0)
        dirty = True
    elif k == ord("f"):
        det.flip, dirty = not det.flip, True
    elif k == ord("b"):
        box3d = not box3d
        tracker, snap, tracked_n = new_tracker(), None, -1
        dirty = True
    elif k == ord("p"):
        show_pts = not show_pts
    elif k == ord("c"):
        p = os.path.join(out_dir, f"frame_{n:03d}.png")
        cv2.imwrite(p, frame)
        print("tersimpan:", p)
    elif paused and k in (ord("a"), ord("d")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 2) if k == ord("a") else n)
        ok, f = cap.read()
        if ok:
            n = n + 1 if k == ord("d") else max(1, n - 1)
            frame, dirty = cv2.resize(f, (1280, 720)), True
            if k == ord("a"):                  # mundur: riwayat tracking tidak berlaku lagi
                tracker, snap, tracked_n = new_tracker(), None, -1
cap.release()
cv2.destroyAllWindows()