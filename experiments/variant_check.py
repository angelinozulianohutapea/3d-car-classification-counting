import argparse, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
import detector as D
from detector import Detector

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="data/video.mp4")
ap.add_argument("--model", default="centernet-3d-bbox.pth")
ap.add_argument("--frame", type=int, default=120)
ap.add_argument("--image", default=None)
a = ap.parse_args()

if a.image:
    f = cv2.imread(a.image)
    if f is None:
        raise SystemExit(f"Gambar tidak terbaca: {a.image}")
else:
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, f = cap.read()
    if not ok:
        raise SystemExit("Frame tidak terbaca")
f = cv2.resize(f, (1280, 720))
W = 1280

def gamma(img, g):
    lut = ((np.arange(256) / 255.0) ** (1.0 / g) * 255).astype("uint8")
    return cv2.LUT(img, lut)

det = Detector(a.model, conf=0.05)
variants = [
    ("asli", 1.0, False, 1.0),
    ("gamma2.0", 2.0, False, 1.0),
    ("flip", 1.0, True, 1.0),
    ("skala1.5", 1.0, False, 1.5),
    ("skala2.0", 1.0, False, 2.0),
    ("gamma2+flip+skala1.5", 2.0, True, 1.5),
]

tiles = []
for name, g, flip, sc in variants:
    img = gamma(f, g) if g != 1.0 else f
    if flip:
        img = cv2.flip(img, 1)
    hm, off, _ = det.infer(img, sc)
    pts = D.decode_points(hm, off, 0.05, cfg.MODEL_SCALE / sc)
    if flip:
        for p in pts:
            p["x"] = W - p["x"]
    pts.sort(key=lambda p: -p["score"])
    print(f"\n[{name}] puncak >=0.05: {len(pts)}")
    for p in pts[:10]:
        print(f'   x={p["x"]:6.0f} y={p["y"]:6.0f} skor={p["score"]:.3f}')
    vis = f.copy()
    for p in pts:
        c = (0, 255, 0) if p["score"] >= 0.15 else (0, 0, 255)
        cv2.circle(vis, (int(p["x"]), int(p["y"])), 7, c, -1)
        cv2.putText(vis, f'{p["score"]:.2f}', (int(p["x"]) + 8, int(p["y"])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    vis = cv2.resize(vis, (640, 360))
    cv2.putText(vis, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    tiles.append(vis)

mosaic = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
cv2.imwrite("experiments/out/variant_check.jpg", mosaic)
print("\ntersimpan: experiments/out/variant_check.jpg")
