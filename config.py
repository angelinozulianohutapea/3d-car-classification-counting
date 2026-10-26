"""
Konfigurasi terpusat. Semua angka yang memengaruhi hasil ada di sini.

Kolom "Dasar" di komentar menyatakan asal nilai:
  [rujukan]    ada dasar literatur (disebut di komentar)
  [kode asal]  diwarisi dari proyek asal antoinekeller/3d_bbox_highway
  [heuristik]  dipilih/diuji manual pada video uji; BELUM ada rujukan
               literatur. Nilainya perlu dievaluasi, jangan dianggap baku.
"""

# ----------------------------------------------------------------------
# Citra dan kalibrasi kamera
# Model kamera lubang jarum (pinhole): Hartley & Zisserman, "Multiple View
# Geometry in Computer Vision", 2nd ed., Cambridge Univ. Press, 2004, bab 6.
# Nilai awal berasal dari simulator CARLA [kode asal]; untuk video nyata
# setel ulang dengan: python -m tools.calibrate_tuner VIDEO.mp4
# ----------------------------------------------------------------------
IMG_W = 1280
IMG_H = 720
PITCH_DEG = 9.80       # kamera menunduk (derajat)  [BELUM disetel untuk video nyata]
CAM_HEIGHT_M = 8.30    # tinggi kamera dari jalan (meter)
FOCAL_PX = 770.00      # focal length (piksel)

# ----------------------------------------------------------------------
# Detektor CenterNet
# Zhou, Wang, Krahenbuhl, "Objects as Points", arXiv:1904.07850, 2019.
# Stride 32 karena backbone ResNet-18 dipotong di layer4 [kode asal].
# ----------------------------------------------------------------------
MODEL_SCALE = 32
DEFAULT_CONF = 0.5         # ambang skor deteksi [kode asal]
CACHE_CONF_DEFAULT = 0.15  # ambang terendah saat membuat cache [heuristik]; samakan dengan CONF
                           # agar cache memuat semua deteksi yang terlihat di live preview

# ----------------------------------------------------------------------
# Filter deteksi (setelah diproyeksikan ke bidang jalan)
# [kode asal] untuk struktur filter; angka ambang [heuristik]
# ----------------------------------------------------------------------
# Jalur valid (meter, x dunia: 0 = tengah gambar, negatif = kiri). Dua arah:
# sisi kanan gambar = kendaraan MENDEKAT, sisi kiri = MENJAUH (jalan tol terpisah median).
# Deteksi di luar [LANE_X_MIN, LANE_X_MAX] dibuang. UKUR batasnya dengan
# python -m tools.box_tuner (klik kendaraan terluar, baca x). Nilai kiri hanya perkiraan.
LANE_X_MIN = -30.0   # hasil sweep: -30 menangkap kendaraan jalur terluar, -34 tidak menambah apa pun
LANE_X_MAX = 11.0
# Median pemisah dua arah (x dunia). x < MEDIAN_X = jalur kiri (MENJAUH), selebihnya MENDEKAT.
# Arah ditentukan dari sisi median, bukan dari gerak per frame, karena gerak per frame berisik.
MEDIAN_X = -8.0
MIN_WIDTH_M = 0.6          # sebelumnya 1.0; dilonggarkan karena lebar dari model CARLA sering meleset
MAX_WIDTH_M = 2.5          # lebar dipotong sampai batas ini
HEIGHT_RANGE_M = (1.0, 5.0)
MIN_LENGTH_M = 3.0

# ----------------------------------------------------------------------
# Kalman Filter (constant velocity pada sumbu jalan)
# R. E. Kalman, "A New Approach to Linear Filtering and Prediction
# Problems", J. Basic Engineering 82(1), 1960.
# Model constant velocity dan bentuk Joseph: Bar-Shalom, Li, Kirubarajan,
# "Estimation with Applications to Tracking and Navigation", Wiley, 2001.
# Besaran Q dan R di bawah ini [heuristik].
# ----------------------------------------------------------------------
MAX_ACCEL = 2.0            # m/s^2, percepatan wajar mobil di jalan tol
LATERAL_STD = 1.0          # m/s, gerak lateral kecil
DIM_STD_PER_STEP = 0.01    # m per frame; dimensi dibuat hampir konstan
# simpangan baku awal state: x, y, vy, w, h, l
KF_INIT_STD = (1.0, 3.0, 20.0, 2.0, 2.0, 3.0)
# Noise pengukuran membesar dengan jarak (depth, meter): std = a + b * depth^p
KF_STD_X = (0.15, 0.010, 1)
KF_STD_Y = (0.20, 0.0005, 2)
KF_STD_DIM = (0.30, 0.015, 1)

# ----------------------------------------------------------------------
# Tracking
# Asosiasi Hungarian: H. W. Kuhn, Naval Research Logistics Quarterly 2(1-2),
# 1955; implementasi SciPy mengikuti D. F. Crouse, IEEE Trans. Aerospace and
# Electronic Systems 52(4), 2016.
# Kombinasi Kalman + Hungarian + min_hits/max_age: A. Bewley dkk., "Simple
# Online and Realtime Tracking" (SORT), ICIP 2016, arXiv:1602.00763.
# Asosiasi dua tahap (--conf-low): Y. Zhang dkk., "ByteTrack", ECCV 2022,
# arXiv:2110.06864.
# Nilai angka di bawah [heuristik].
# ----------------------------------------------------------------------
MAX_LOST_FRAMES = 30        # track dihapus setelah sekian frame tanpa deteksi (max_age)
MIN_HITS_TO_DISPLAY = 3     # track baru dipakai/ditampilkan setelah sekian deteksi (min_hits)
MAX_LOST_TO_DISPLAY = 5     # track hilang > sekian frame tidak ditampilkan
MAX_TRACK_DEPTH_M = 40.0    # track lebih jauh dari ini dihapus
CONFLICT_DIST_M = 3.0       # dua track lebih dekat dari ini dianggap duplikat
# Gerbang asosiasi adaptif: batas = base + slope * kedalaman (meter).
# Objek jauh lebih noisy, objek yang baru hilang diberi toleransi lebih besar.
GATE_TRACKED = {"dist": (2.0, 0.05), "lateral": (1.0, 0.01)}
GATE_LOST = {"dist": (4.0, 0.05), "lateral": (2.0, 0.01)}

# ----------------------------------------------------------------------
# Klasifikasi kendaraan (dari ukuran terukur, dengan voting)
# SELURUH bagian ini [heuristik]: ambang panjang dan batas ukuran dipilih
# manual. Detektor tidak membedakan jenis kendaraan (head heatmap 1 kanal).
# ----------------------------------------------------------------------
CLASSES = ("car", "van", "truck")   # urutan = prioritas saat suara seri
CAR_MAX_LEN_M = 5.2
VAN_MAX_LEN_M = 7.0
TEMPLATES = {
    "car": {"w": 1.85, "h": 1.50, "l": 4.50},
    "van": {"w": 2.00, "h": 2.00, "l": 5.60},
    "truck": {"w": 2.50, "h": 3.20, "l": 9.00},
}
LIMITS = {
    "car": {"w": (1.6, 2.1), "h": (1.2, 1.9), "l": (3.5, 5.3)},
    "van": {"w": (1.8, 2.3), "h": (1.7, 2.6), "l": (4.8, 7.0)},
    "truck": {"w": (2.2, 2.6), "h": (2.5, 4.2), "l": (6.5, 13.0)},
}
DIM_BUF_SIZE = 30          # jumlah pengamatan dimensi yang disimpan (median)

# ----------------------------------------------------------------------
# Kotak 3D untuk tampilan [heuristik]
# ----------------------------------------------------------------------
MEASURED_WEIGHT = 0.5      # 0 = template saja, 1 = ukuran terukur saja
DIM_WARMUP = 10            # bobot ukuran terukur naik sampai sekian pengamatan
YAW_WINDOW = 10            # jumlah posisi terakhir untuk menaksir arah
YAW_MIN_DISPLACEMENT_M = 1.0
YAW_ALPHA = 0.2            # EMA pada sudut ganda
YAW_MAX_DEG = 35.0

# ----------------------------------------------------------------------
# Counting dengan garis virtual [heuristik]
# ----------------------------------------------------------------------
LINE_Y_DEFAULT_M = 15.0    # garis untuk kendaraan yang MENDEKAT (meter dari kamera)
# Garis untuk kendaraan yang MENJAUH. Harus lebih jauh dari tempat track mulai
# terdeteksi (bawah gambar sekitar 12 m), kalau tidak track lahir setelah garis dan tak terhitung.
LINE_Y_AWAY_DEFAULT_M = 25.0
# Track harus sudah terlihat minimal sekian meter SEBELUM garis agar boleh dihitung.
# Mencegah track yang baru lahir dekat garis (ID putus lalu dibuat ulang) ikut terhitung.
COUNT_MIN_PRE_M = 4.0
# Jalur kiri (menjauh): dihitung setelah bergerak sejauh ini (m) dari tempat pertama terlihat.
AWAY_MIN_TRAVEL_M = 2.0
COUNT_LOG = "counting_log.csv"
OUTPUT_DIR = "output_frames"

# ----------------------------------------------------------------------
# Akurasi deteksi (dipakai detector.py) [heuristik, nilai awal untuk diuji]
# ----------------------------------------------------------------------
CONF = 0.15            # ambang skor; DEFAULT_CONF=0.5 di atas terlalu ketat
SCALES = (1.0,)       # skala input; (1.0, 1.5) membantu mobil kecil/jauh tetapi ~3,3x lebih lambat
GAMMA = 2.0            # >1 mencerahkan area gelap (mobil hitam); 1.0 = mati
FLIP = False           # True = tambah pass flip horizontal (2x lebih lambat)
MERGE_DIST_PX = 48.0   # jarak piksel untuk menggabungkan deteksi duplikat