"""
Counting dengan garis virtual.

Garis berada pada kedalaman `line_y` (meter dari kamera) dan membentang selebar
jalur valid. Sebuah track dianggap melintas saat posisi y-nya berpindah dari
satu sisi garis ke sisi lain. Dua arah dihitung: sisi median menentukan arahnya
(x < MEDIAN_X = menjauh, memakai line_y_away). Setiap track dihitung
paling banyak sekali, dengan kelas hasil voting saat itu, dan hanya setelah
track layak tampil (min_hits), supaya deteksi palsu sesaat tidak terhitung.

Dasar: counting dengan garis virtual adalah praktik umum; bentuk aturan ini
[heuristik proyek] dan belum divalidasi terhadap hitungan manual.
"""

import csv

import config as cfg

HEADER = ["frame", "id", "kelas", "panjang_m", "lebar_m", "tinggi_m", "x_m"] + [
    f"suara_{c}" for c in cfg.CLASSES
] + ["arah"]

# arah = sisi median saat melintasi garis (x < MEDIAN_X = menjauh)
DIRECTIONS = ("mendekat", "menjauh")


class _State:
    __slots__ = ("prev_y", "crossed", "counted", "direction", "y_first")

    def __init__(self):
        self.prev_y = None
        self.crossed = False
        self.counted = False
        self.direction = None
        self.y_first = None


class LineCounter:
    def __init__(self, line_y=cfg.LINE_Y_DEFAULT_M, line_y_away=None):
        """line_y: garis untuk kendaraan yang mendekat; line_y_away: garis untuk yang
        menjauh (None = sama dengan line_y)."""
        self.line_y = line_y
        self.line_y_away = line_y if line_y_away is None else line_y_away
        self.counts = {c: 0 for c in cfg.CLASSES}
        self.dir_counts = {d: {c: 0 for c in cfg.CLASSES} for d in DIRECTIONS}
        self.events = []    # satu baris per kendaraan yang dihitung (format HEADER)
        self.counted_at = {}  # track.id -> (frame, arah), untuk penanda visual
        self.history = {}   # display_id -> ringkasan hidup track yang pernah tampil
        self._state = {}    # track.id -> _State

    def total(self):
        return sum(self.counts.values())

    def update(self, tracks, frame_idx):
        """Panggil sekali per frame, setelah Tracker.assign_display_ids()."""
        for t in tracks:
            st = self._state.setdefault(t.id, _State())
            y = t.y()
            prev, st.prev_y = st.prev_y, y
            if st.y_first is None:
                st.y_first = y

            # Lintasan dicatat walau track sedang tidak terlihat, supaya tidak hilang.
            # Arah dari sisi median; tiap arah punya garisnya sendiri.
            away = t.x() < cfg.MEDIAN_X
            line = self.line_y_away if away else self.line_y
            if away:
                # Jalur kiri: kendaraan masuk dari tepi kiri gambar, jadi tempat lahirnya
                # bergantung pada x dan garis tetap tidak cocok. Dihitung bila sudah
                # bergerak menjauh minimal AWAY_MIN_TRAVEL_M dari tempat pertama terlihat.
                if not st.crossed and y - st.y_first >= cfg.AWAY_MIN_TRAVEL_M:
                    st.crossed = True
                    st.direction = "menjauh"
            else:
                m = cfg.COUNT_MIN_PRE_M
                stable = st.y_first >= line + m
                if prev is not None and stable and not st.crossed and prev != y and (prev - line) * (y - line) <= 0:
                    st.crossed = True
                    st.direction = "mendekat"

            if st.crossed and not st.counted and t.display_id is not None:
                st.counted = True
                self._count(t, frame_idx, st.direction)

            if t.display_id is not None:
                self._record(t, st, y, frame_idx)

    def _count(self, t, frame_idx, direction):
        cls = t.cls
        self.counts[cls] += 1
        self.dir_counts[direction][cls] += 1
        self.counted_at[t.id] = (frame_idx, direction)
        w, h, l = t.classifier.median_dims()
        votes = t.classifier.votes
        self.events.append(
            [
                frame_idx,
                t.display_id,
                cls,
                round(float(l), 2),
                round(float(w), 2),
                round(float(h), 2),
                round(float(t.x()), 2),
            ]
            + [votes[c] for c in cfg.CLASSES]
            + [direction]
        )

    def _record(self, t, st, y, frame_idx):
        rec = self.history.setdefault(
            t.display_id,
            {"first": frame_idx, "y_first": y, "y_min": y, "y_max": y, "x_first": t.x()},
        )
        rec["last"] = frame_idx
        rec["y_last"] = y
        rec["y_min"] = min(rec["y_min"], y)
        rec["y_max"] = max(rec["y_max"], y)
        rec["cls"] = t.cls
        rec["counted"] = st.counted

    # ------------------------------------------------------------------
    def save_log(self, path=cfg.COUNT_LOG):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(self.events)

    def print_report(self, log_path=cfg.COUNT_LOG):
        print(f"\nHasil counting (disimpan di {log_path}):")
        for c in cfg.CLASSES:
            print(f"  {c:6}: {self.counts[c]}")
        print(f"  total : {self.total()}")
        print("\nPer arah:")
        for d in DIRECTIONS:
            row = self.dir_counts[d]
            detail = ", ".join(f"{c} {row[c]}" for c in cfg.CLASSES)
            print(f"  {d:8}: {sum(row.values()):3}  ({detail})")

        uncounted = {k: v for k, v in self.history.items() if not v["counted"]}
        print(
            f"\nTrack yang tampil: {len(self.history)}, "
            f"dihitung: {len(self.history) - len(uncounted)}, "
            f"tidak dihitung: {len(uncounted)}"
        )
        if uncounted:
            print("Track tampil tapi tidak dihitung:")
            print("  id   kelas   frame (awal-akhir)   y awal -> y akhir (m)   x awal   keterangan")
            for k in sorted(uncounted):
                v = uncounted[k]
                if v["y_min"] > max(self.line_y, self.line_y_away):
                    note = "tidak pernah sampai garis (selalu lebih jauh dari garis)"
                elif v["y_max"] < min(self.line_y, self.line_y_away):
                    note = "lahir setelah garis (selalu lebih dekat dari garis)"
                elif abs(v["y_first"] - (self.line_y_away if v["x_first"] < cfg.MEDIAN_X else self.line_y)) < cfg.COUNT_MIN_PRE_M:
                    note = "lahir terlalu dekat garis (tidak dihitung)"
                else:
                    note = "melewati garis tapi tidak terhitung (cek)"
                print(
                    f"  {k:3}  {v['cls']:6}  {v['first']:5}-{v['last']:<5}"
                    f"  {v['y_first']:6.1f} -> {v['y_last']:6.1f}  {v['x_first']:6.1f}   {note}"
                )

        print("\nDetail tiap kendaraan yang dihitung:")
        print("  frame   id  kelas   panjang  lebar  tinggi  x(m)  suara(car/van/truck)  arah")
        n_cls = len(cfg.CLASSES)
        for e in self.events:
            print(
                f"  {e[0]:5}  {e[1]:3}  {e[2]:6}  {e[3]:6.2f}  {e[4]:5.2f}  {e[5]:6.2f}"
                f"  {e[6]:5.1f}  {'/'.join(str(v) for v in e[7:7 + n_cls])}  {e[-1]}"
            )
