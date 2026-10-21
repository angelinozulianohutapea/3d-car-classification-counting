"""
Cache deteksi mentah, supaya detektor cukup dijalankan sekali dan percobaan
tracking berikutnya jauh lebih cepat. Format pickle (kompatibel dengan file
.pkl lama): {"nms": "pool", "min_conf": float, "frames": [daftar box per frame]}.

PERINGATAN: pickle dapat menjalankan kode saat dimuat. Hanya muat file yang
Anda buat sendiri.
"""

import pickle


def load(path, min_conf):
    with open(path, "rb") as f:
        cache = pickle.load(f)
    if cache.get("nms", "pool") != "pool":
        raise SystemExit(
            f"Cache dibuat dengan NMS '{cache['nms']}', sedangkan kode ini hanya "
            "memakai 'pool'. Buat ulang cache."
        )
    if min_conf < cache["min_conf"] - 1e-9:
        raise SystemExit(
            f"Ambang {min_conf} lebih rendah dari ambang cache ({cache['min_conf']}). "
            "Buat ulang cache dengan --cache-conf yang lebih rendah."
        )
    return cache["frames"]


def load_frames(path):
    """Hanya daftar box per frame (untuk alat pelabelan)."""
    with open(path, "rb") as f:
        return pickle.load(f)["frames"]


def save(path, frames, min_conf):
    with open(path, "wb") as f:
        pickle.dump({"nms": "pool", "min_conf": min_conf, "frames": frames}, f)
