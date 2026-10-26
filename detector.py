"""
Deteksi kendaraan dengan CenterNet versi sederhana (ResNet-18, stride 32).

Keluaran model per frame (peta ~23x40 untuk input 1280x720):
  heatmap [1,H,W], offset [2,H,W], ukuran w/h/l [3,H,W].
Puncak heatmap diambil dengan max-pooling 3x3 sebagai pengganti NMS.

Rujukan: X. Zhou, D. Wang, P. Krahenbuhl, "Objects as Points",
arXiv:1904.07850, 2019, bagian 4 (peak extraction). Arsitektur dan bobot
awal: proyek antoinekeller/3d_bbox_highway [kode asal].
Head heatmap hanya 1 kanal, jadi detektor TIDAK membedakan jenis kendaraan.

Perbaikan akurasi [heuristik, perlu dievaluasi pada video sendiri]:
  - ukuran negatif di-clamp (tidak lagi membuang deteksi)
  - koreksi gamma (mengangkat area gelap, mis. mobil hitam)
  - inferensi multi-skala, opsional flip horizontal (test-time augmentation)
  - penggabungan hasil dengan NMS berbasis jarak
  - ambang dan opsi dibaca dari config.py (CONF, SCALES, GAMMA, FLIP, MERGE_DIST_PX)
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import transforms

import config as cfg

MIN_SIZE = 1e-3  # batas bawah w/h/l setelah clamp


class CenterNet(nn.Module):
    """ResNet-18 sampai layer4 (512 kanal, stride 32) + 3 head."""

    def __init__(self):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        self.base_model = nn.Sequential(*list(base.children())[:-2])

        def head(out_ch, num_ch=512, head_conv=64):
            return nn.Sequential(
                nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, out_ch, kernel_size=1, stride=1),
            )

        self.outc = head(1)
        self.outo = head(2)
        self.outr = head(3)

    def forward(self, x):
        x = self.base_model(x)
        return self.outc(x), self.outo(x), self.outr(x)


def load_model(path, device="cpu"):
    model = CenterNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(device)
    model.eval()
    return model


def build_preprocess():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


_LUT = {}


def apply_gamma(img, g):
    """g > 1 mencerahkan area gelap. g == 1 tidak mengubah apa pun."""
    if g == 1.0:
        return img
    lut = _LUT.get(g)
    if lut is None:
        lut = _LUT[g] = ((np.arange(256) / 255.0) ** (1.0 / g) * 255).astype("uint8")
    return cv2.LUT(img, lut)


def select_pool(hm):
    t = torch.from_numpy(np.ascontiguousarray(hm)).unsqueeze(0).unsqueeze(0)
    keep = F.max_pool2d(t, kernel_size=3, stride=1, padding=1) == t
    return hm * keep.squeeze(0).squeeze(0).numpy()


def decode_points(hm, offset, conf, stride=None):
    """Puncak heatmap > conf sebagai dict {x, y (piksel), score, cy, cx}."""
    stride = cfg.MODEL_SCALE if stride is None else stride
    peaks = select_pool(hm)
    out = []
    for cy, cx in np.argwhere(peaks > conf):
        out.append(
            {
                "x": float((cx + offset[0, cy, cx]) * stride),
                "y": float((cy + offset[1, cy, cx]) * stride),
                "score": float(peaks[cy, cx]),
                "cy": int(cy),
                "cx": int(cx),
            }
        )
    return out


def decode_boxes(hm, offset, whl, conf, stride=None):
    """Seperti decode_points, ditambah w, h, l (negatif di-clamp, bukan dibuang)."""
    boxes = []
    for p in decode_points(hm, offset, conf, stride):
        w, h, l = (max(float(v), MIN_SIZE) for v in whl[:, p["cy"], p["cx"]])
        boxes.append({"x": p["x"], "y": p["y"], "w": w, "h": h, "l": l, "score": p["score"]})
    return boxes


def merge_boxes(boxes, dist=48.0):
    """NMS berbasis jarak pusat: skor tertinggi menang."""
    boxes = sorted(boxes, key=lambda b: b["score"], reverse=True)
    kept = []
    for b in boxes:
        if all(np.hypot(b["x"] - k["x"], b["y"] - k["y"]) >= dist for k in kept):
            kept.append(b)
    return kept


class Detector:
    def __init__(self, model_path, conf=None, scales=None, gamma=None, flip=None,
                 merge_dist=None, rgb=False, debug=False):
        self.conf = conf if conf is not None else getattr(cfg, "CONF", 0.15)
        self.scales = tuple(scales) if scales is not None else tuple(getattr(cfg, "SCALES", (1.0,)))
        self.gamma = gamma if gamma is not None else getattr(cfg, "GAMMA", 1.0)
        self.flip = flip if flip is not None else getattr(cfg, "FLIP", False)
        self.merge_dist = merge_dist if merge_dist is not None else getattr(cfg, "MERGE_DIST_PX", 48.0)
        self.rgb = rgb
        self.debug = debug
        self.last_peaks = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(2)
        print(f"Device: {self.device} | conf={self.conf} | scales={self.scales} "
              f"| gamma={self.gamma} | flip={self.flip}")
        self.model = load_model(model_path, self.device)
        self.preprocess = build_preprocess()

    def infer(self, frame, scale=1.0):
        """Frame BGR -> (heatmap [H,W] sigmoid, offset [2,H,W], whl [3,H,W])."""
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if self.rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = self.preprocess(frame).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            hm, offset, whl = self.model(x)
        hm = torch.sigmoid(hm).cpu().numpy()[0, 0]
        return hm, offset.cpu().numpy()[0], whl.cpu().numpy()[0]

    def detect(self, frame):
        """Frame BGR -> daftar dict {x, y (piksel), w, h, l, score}."""
        width = frame.shape[1]
        img = apply_gamma(frame, self.gamma)
        flips = (False, True) if self.flip else (False,)
        boxes = []
        self.last_peaks = []
        for fl in flips:
            im = cv2.flip(img, 1) if fl else img
            for s in self.scales:
                hm, offset, whl = self.infer(im, s)
                stride = cfg.MODEL_SCALE / s
                found = decode_boxes(hm, offset, whl, self.conf, stride)
                if self.debug:
                    self.last_peaks += [
                        (s, (width - p["x"]) if fl else p["x"], p["y"], p["score"])
                        for p in decode_points(hm, offset, 0.05, stride)
                    ]
                if fl:
                    for b in found:
                        b["x"] = width - b["x"]
                boxes += found
        if len(self.scales) > 1 or self.flip:
            boxes = merge_boxes(boxes, self.merge_dist)
        return boxes