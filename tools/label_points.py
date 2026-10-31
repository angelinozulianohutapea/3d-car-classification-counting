"""
Alat pelabelan titik mobil (satu klik per mobil), disimpan ke labels.json.

Klik kiri  : tambah titik (titik kontak mobil dengan jalan)
Klik kanan : hapus titik terdekat
d / a      : frame +1 / -1        w / s : frame +5 / -5
u          : batalkan titik terakhir
1 / 2 / 3  : kelas titik terakhir = car / van / truck (0 = hapus kelas); opsional
SPASI      : tandai frame SELESAI (boleh 0 titik = contoh negatif)
m          : tampil/sembunyi titik deteksi model dari cache (kuning)
q / ESC    : simpan dan keluar

Penting: pada frame yang ditandai SELESAI, SEMUA mobil harus diberi titik,
karena mobil tanpa titik dianggap latar saat pelatihan.

Format labels.json per frame: {"done": bool, "points": [[x, y], ...], "kelas": [...]}.
"kelas" sejajar dengan "points" (None bila belum diisi) dan opsional: berkas lama tanpa
kelas tetap terbaca. Kelas belum dipakai pelatihan; fungsinya sebagai kebenaran
dasar untuk mengevaluasi klasifikasi car/van/truck.

Pakai (dari folder proyek):
  python -m tools.label_points VIDEO.mp4 --out labels.json --cache det_low.pkl
"""

import json
import os
from argparse import ArgumentParser

import cv2

import detection_cache

CLASSES = {ord("1"): "car", ord("2"): "van", ord("3"): "truck"}


def main():
    ap = ArgumentParser(description=__doc__.split("\n")[1])
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
    cache_frames = detection_cache.load_frames(args.cache) if args.cache else None

    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    state = {"idx": min(max(args.start, 0), n - 1), "cache": True}

    def rec(i):
        return data.setdefault(str(i), {"done": False, "points": []})

    def kelas_of(r):
        """Daftar kelas sejajar dengan points (tanpa mengubah r); None = belum diisi."""
        k = list(r.get("kelas", []))[: len(r["points"])]
        return k + [None] * (len(r["points"]) - len(k))

    def sync_kelas(r):
        """Pastikan r['kelas'] ada dan sejajar dengan r['points'] sebelum diubah."""
        r["kelas"] = kelas_of(r)
        return r["kelas"]

    def save():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def on_mouse(event, x, y, flags, param):
        x, y = x / args.scale, y / args.scale
        r = rec(state["idx"])
        if event == cv2.EVENT_LBUTTONDOWN:
            sync_kelas(r).append(None)
            r["points"].append([round(x, 1), round(y, 1)])
            save()
        elif event == cv2.EVENT_RBUTTONDOWN and r["points"]:
            j = min(range(len(r["points"])),
                    key=lambda k: (r["points"][k][0] - x) ** 2 + (r["points"][k][1] - y) ** 2)
            sync_kelas(r).pop(j)
            r["points"].pop(j)
            save()

    cv2.namedWindow("label")
    cv2.setMouseCallback("label", on_mouse)

    while True:
        i = state["idx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        r = rec(i)

        if state["cache"] and cache_frames is not None:
            for b in cache_frames[i]:
                cv2.circle(frame, (int(b["x"]), int(b["y"])), 4, (0, 255, 255), -1)
                cv2.putText(frame, f"{b['score']:.2f}", (int(b["x"]) + 6, int(b["y"]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        kls = kelas_of(r)
        for k, (px, py) in enumerate(r["points"]):
            cv2.circle(frame, (int(px), int(py)), 7, (0, 0, 255), -1)
            cv2.putText(frame, f"{k + 1}" + (f" {kls[k]}" if kls[k] else ""),
                        (int(px) + 9, int(py) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        n_done = sum(1 for v in data.values() if v["done"])
        status = "SELESAI" if r["done"] else "belum"
        cv2.putText(frame, f"frame {i}/{n - 1} | titik {len(r['points'])} (berkelas {sum(1 for c in kls if c)}) | {status} | frame selesai: {n_done}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if r["done"] else (255, 255, 255), 2)
        shown = frame if args.scale == 1.0 else cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        cv2.imshow("label", shown)

        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("d"):
            state["idx"] = min(i + 1, n - 1)
        elif k == ord("a"):
            state["idx"] = max(i - 1, 0)
        elif k == ord("w"):
            state["idx"] = min(i + 5, n - 1)
        elif k == ord("s"):
            state["idx"] = max(i - 5, 0)
        elif k == ord("u") and r["points"]:
            sync_kelas(r).pop()
            r["points"].pop()
            save()
        elif (k in CLASSES or k == ord("0")) and r["points"]:
            sync_kelas(r)[-1] = CLASSES.get(k)
            save()
        elif k == 32:
            r["done"] = not r["done"]
            save()
        elif k == ord("m"):
            state["cache"] = not state["cache"]

    save()
    cap.release()
    cv2.destroyAllWindows()
    total = sum(len(v["points"]) for v in data.values() if v["done"])
    print(f"Tersimpan di {args.out}: {sum(1 for v in data.values() if v['done'])} frame selesai, {total} titik")
    hitung = {}
    for v in data.values():
        if v["done"]:
            for c in kelas_of(v):
                hitung[c or "belum diberi kelas"] = hitung.get(c or "belum diberi kelas", 0) + 1
    print("Per kelas:", ", ".join(f"{k} {n}" for k, n in sorted(hitung.items())))


if __name__ == "__main__":
    main()