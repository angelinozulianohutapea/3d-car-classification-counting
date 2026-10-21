"""
Klasifikasi kendaraan: car / van / truck.

Kelas ditentukan dari PANJANG terukur (hasil proyeksi ke bidang jalan), bukan
dari tampilan gambar, karena head heatmap detektor hanya 1 kanal (tidak
membedakan jenis kendaraan). Setiap pengamatan memberi satu suara, kelas akhir
= suara terbanyak (seri dimenangkan kelas yang lebih awal di config.CLASSES).

Dasar: ambang panjang dan voting adalah [heuristik proyek], belum ada rujukan.
Hasil akan keliru bila panjang terukur meleset (misalnya sedan terbaca truck).
"""

from collections import deque

import numpy as np

import config as cfg


def classify_by_length(length):
    if length < cfg.CAR_MAX_LEN_M:
        return "car"
    if length < cfg.VAN_MAX_LEN_M:
        return "van"
    return "truck"


class VehicleClassifier:
    def __init__(self):
        self.votes = {c: 0 for c in cfg.CLASSES}
        self.cls = cfg.CLASSES[0]
        self.dim_buf = deque(maxlen=cfg.DIM_BUF_SIZE)  # (w, h, l) terukur

    def update(self, w, h, l):
        """Dipanggil setiap kali track mendapat deteksi."""
        self.votes[classify_by_length(l)] += 1
        self.cls = max(self.votes, key=self.votes.get)
        self.dim_buf.append((w, h, l))

    def median_dims(self):
        """Median (w, h, l) terukur."""
        return np.median(np.array(self.dim_buf), axis=0)
