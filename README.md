# Deteksi, Tracking, Klasifikasi, dan Counting Kendaraan dengan Kotak 3D

Sistem untuk video jalan raya dari satu kamera statis (misalnya di jembatan penyeberangan). Empat fungsi utama:

| Fungsi | Modul | Metode | Rujukan |
| --- | --- | --- | --- |
| Deteksi | `detector.py`, `detection_filter.py` | CenterNet (ResNet-18) menghasilkan titik pusat dan ukuran; puncak heatmap dengan max-pooling 3x3; titik diproyeksikan ke bidang jalan | Zhou dkk. 2019; Hartley & Zisserman 2004 |
| Tracking | `tracker.py`, `kalman_filter.py` | Kalman Filter constant velocity + asosiasi Hungarian dengan gerbang adaptif; opsional dua tahap (`--conf-low`) | Kalman 1960; Kuhn 1955; Bewley dkk. 2016; Zhang dkk. 2022 |
| Klasifikasi | `classifier.py` | car / van / truck dari panjang terukur, dengan voting antar frame | heuristik proyek (belum ada rujukan) |
| Counting | `counter.py` | garis virtual; satu track dihitung sekali; log `counting_log.csv` | heuristik proyek (belum ada rujukan) |

Kotak 3D berorientasi (`box3d.py`, `visualize.py`) hanya untuk tampilan dan tidak memengaruhi hasil keempat fungsi di atas.

## Instalasi

Python 3.10 atau lebih baru (diuji di Python 3.12, Ubuntu).

```
git clone https://github.com/angelinozulianohutapea/3d-car-classification-counting.git
cd 3d-car-classification-counting
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Model (`.pth`) dan video tidak disertakan di repositori. Video harus berukuran **1280x720** (kalibrasi dan detektor mengasumsikan ukuran ini; program berhenti dengan pesan jelas bila ukurannya lain).

- Model CenterNet: `<isi tautan unduhan atau cara melatih>`
- Video contoh: `<isi sumber video>`

## Struktur

```
config.py            semua konstanta (kalibrasi, ambang, parameter tracking) beserta dasar nilainya
geometry.py          model kamera, proyeksi 3D <-> 2D
detector.py          CenterNet dan ekstraksi puncak heatmap
detection_cache.py   simpan/muat deteksi mentah
detection_filter.py  proyeksi ke jalan dan filter deteksi
kalman_filter.py     Kalman Filter
tracker.py           Track dan Tracker (asosiasi, siklus hidup track)
classifier.py        klasifikasi car/van/truck
counter.py           garis virtual dan log CSV
box3d.py             dimensi dan arah kotak 3D (tampilan)
visualize.py         menggambar hasil
run.py               program utama
tools/               kalibrasi, pelabelan, fine-tuning, evaluasi
tests/               uji sintetis (tanpa video dan tanpa PyTorch)
```

## Cara pakai

### 1. Kalibrasi kamera

Nilai bawaan (pitch 9,8 derajat, tinggi 8,30 m, focal 770 px) berasal dari simulator CARLA, jadi untuk video nyata perlu disetel ulang:

```
python -m tools.calibrate_tuner VIDEO.mp4 --frame 100
```

Setel `PITCH` sampai garis horizon pas, lalu `FOCAL` dan `HEIGHT` sampai grid searah marka jalan, geser `GRID_OX`, lalu letakkan kotak hijau (`REF_X`, `REF_Z`) di atas satu mobil. Tekan `q`; tiga baris yang tercetak (`PITCH_DEG`, `CAM_HEIGHT_M`, `FOCAL_PX`) disalin ke `config.py`. ESC untuk batal.

### 2. Menjalankan

```
python run.py VIDEO.mp4 MODEL.pth
python run.py VIDEO.mp4 MODEL.pth --no-show --no-save --line-y 15
python run.py VIDEO.mp4 MODEL.pth --cache det_low.pkl --conf 0.5 --conf-low 0.3 --no-show --no-save
```

Pada contoh terakhir, jalankan pertama membuat `det_low.pkl`; jalankan berikutnya memuatnya sehingga detektor tidak dijalankan lagi.

| Opsi | Keterangan |
| --- | --- |
| `--conf` | Ambang skor deteksi (default 0,5) |
| `--conf-low` | Aktifkan asosiasi dua tahap: deteksi skor antara `--conf-low` dan `--conf` hanya menyambung track yang ada. Harus > 0 dan < `--conf` |
| `--line-y` | Posisi garis counting, meter dari kamera (default 15) |
| `--cache` | File cache deteksi mentah: ada = dimuat, belum ada = dibuat |
| `--cache-conf` | Ambang skor terendah saat membuat cache (default 0,2) |
| `--no-show` | Tanpa jendela (SSH / VS Code remote) |
| `--no-save` | Jangan simpan PNG tiap frame ke `output_frames/` |
| `--play` | Langsung jalan tanpa menunggu tombol |
| `-f N` | Putar otomatis sampai frame ke-N, setelahnya berhenti tiap frame |
| `--debug` | Tandai deteksi mentah: hijau = lolos, kuning = lolos skor rendah, merah = ditolak (dengan alasan) |

Tombol jendela: `SPASI` play/pause, `ESC` keluar. Parameter lain (batas lebar, jarak duplikat, tinggi kamera, dan seterusnya) ada di `config.py`.

### 3. Keluaran

- `counting_log.csv`: satu baris per kendaraan yang dihitung.

  | Kolom | Isi |
  | --- | --- |
  | `frame` | frame saat dihitung |
  | `id` | ID tampilan |
  | `kelas` | kelas hasil voting |
  | `panjang_m`, `lebar_m`, `tinggi_m` | dimensi terukur (median) |
  | `x_m` | posisi lateral saat melintas |
  | `suara_car`, `suara_van`, `suara_truck` | jumlah suara tiap kelas |

- Ringkasan di terminal: hitungan per kelas, daftar track yang tampil tetapi tidak dihitung (dengan keterangan), detail tiap kendaraan, dan jumlah deteksi yang dibuang filter per alasan.
- `output_frames/image_XXXX.png`: frame hasil (kecuali `--no-save`).

### Aturan penting

- **Counting**: track dianggap melintas saat posisinya berpindah dari satu sisi garis ke sisi lain (arah tidak dibedakan). Satu track dihitung paling banyak sekali, hanya setelah terdeteksi minimal 3 kali (`MIN_HITS_TO_DISPLAY`).
- **Klasifikasi**: panjang < 5,2 m = car; < 7,0 m = van; selain itu truck. Tiap pengamatan memberi satu suara; kelas akhir adalah suara terbanyak.
- **Filter deteksi**: dibuang bila di luar jalur (\|x\| > 11 m), lebar < 0,6 m, tinggi di luar 1 sampai 5 m, panjang < 3 m, atau tinggi > panjang.

## Pelabelan, fine-tuning, dan evaluasi

Model awal dilatih pada data simulasi CARLA. Untuk menyesuaikannya dengan video nyata, head heatmap dapat dilatih ulang dari label titik (backbone serta head offset dan ukuran dibekukan, tanpa augmentasi). Jalankan dari folder proyek:

```
python -m tools.label_points VIDEO.mp4 --out labels.json --cache det_low.pkl
python -m tools.finetune_heatmap VIDEO.mp4 centernet-3d-bbox.pth --val-from 330
python -m tools.cek_fp VIDEO.mp4 centernet-ft.pth --conf 0.3 --from 330 --to 472 --out cek_ft
python -m tools.lembar_hitungan VIDEO.mp4 --log counting_log.csv --line-y 15
```

`label_points`: klik kiri tambah titik, klik kanan hapus, `d`/`a` frame ±1, `w`/`s` frame ±5, `u` batalkan, `SPASI` tandai SELESAI, `m` tampil/sembunyi titik model, `q`/ESC simpan dan keluar. Pada frame SELESAI, semua mobil harus diberi titik.

**Metrik deteksi** (`tools/metrics.py`, dipakai `finetune_heatmap` dan `cek_fp`): deteksi dianggap benar bila berjarak ≤ 48 piksel dari label yang belum terpakai, dihitung hanya di dalam jalur. Dilaporkan benar (TP), palsu (FP), terlewat (FN), serta precision, recall, dan F1. Karena label berupa titik, IoU dan mAP tidak berlaku. Dekode evaluasi memakai offset model, sama seperti `Detector`.

## Menyetel hasil

| Gejala | Yang disesuaikan |
| --- | --- |
| Kotak masih goyang | naikkan `KF_STD_*` di `config.py` |
| Kotak tertinggal saat mobil mengerem/mempercepat | naikkan `MAX_ACCEL` |
| Ukuran kotak berubah-ubah | turunkan `DIM_STD_PER_STEP` |
| Objek terlambat muncul | turunkan `MIN_HITS_TO_DISPLAY` |
| ID sering berganti untuk mobil yang sama | naikkan `MAX_LOST_FRAMES` atau longgarkan `GATE_*` |
| Mobil hilang-timbul sehingga tidak terhitung | aktifkan `--conf-low` dan/atau turunkan `--conf` |
| Satu mobil menjadi dua track | naikkan `CONFLICT_DIST_M` |
| Deteksi palsu masuk hitungan | naikkan `--conf`; periksa alasan penolakan dengan `--debug` |

## Pengujian

```
python -m tests.test_pipeline
```

Menguji geometri, Kalman Filter, klasifikasi, filter, tracking, dan counting dengan data sintetis (tanpa video dan tanpa PyTorch).

## Bagian tanpa rujukan literatur

Bagian berikut dipilih manual dan diuji pada video uji; nilainya belum divalidasi dan perlu dievaluasi sebelum dijadikan klaim:

- ambang panjang klasifikasi (5,2 m dan 7,0 m), template dan batas ukuran per kelas, serta voting
- bentuk dan angka gerbang asosiasi adaptif, `MAX_LOST_FRAMES`, `MIN_HITS_TO_DISPLAY`, `CONFLICT_DIST_M`
- besaran Q dan R Kalman Filter (modelnya sendiri punya rujukan)
- aturan garis virtual untuk counting
- pencampuran template dan ukuran terukur, serta penghalusan yaw (hanya tampilan)
- toleransi 48 piksel dan `--sigma` 0,8 pada fine-tuning

## Keterbatasan

- Model awal dilatih pada simulasi CARLA, sehingga ada domain gap pada video nyata: mobil bisa tidak terdeteksi, ada deteksi palsu, dan ukuran meleset.
- Detektor tidak membedakan jenis kendaraan. Klasifikasi sepenuhnya bergantung pada panjang hasil kalibrasi, sehingga sedan bisa terbaca truck bila panjangnya meleset. Hanya mobil (3 kelas) yang ditangani; motor, bajaj, becak, dan sepeda belum.
- Fine-tuning hanya melatih head heatmap (posisi pusat); estimasi ukuran masih dari model asli.
- Data latih dan validasi berasal dari satu video (dibagi menurut nomor frame), jadi hasilnya belum menunjukkan kemampuan pada video lain.
- Kalibrasi disetel manual; pitch, tinggi, dan focal saling memengaruhi.
- Bidang jalan dianggap datar dan kamera statis. Resolusi harus 1280x720.
- Counting tidak membedakan arah, dan track yang belum layak tampil saat melewati garis tidak dihitung. Dua kendaraan yang lebih dekat dari `CONFLICT_DIST_M` bisa dianggap duplikat.
- Metrik tracking (MOTA, IDF1) dan klasifikasi (confusion matrix) belum ada karena belum ada label ID dan kelas.

## Rujukan

- X. Zhou, D. Wang, P. Krähenbühl, "Objects as Points", arXiv:1904.07850, 2019 (CenterNet, ekstraksi puncak, focal loss).
- T.-Y. Lin dkk., "Focal Loss for Dense Object Detection", ICCV 2017.
- Y. Zhang dkk., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022, arXiv:2110.06864.
- A. Bewley dkk., "Simple Online and Realtime Tracking", ICIP 2016, arXiv:1602.00763.
- R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems", J. Basic Engineering 82(1), 1960.
- Y. Bar-Shalom, X. R. Li, T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley, 2001.
- H. W. Kuhn, "The Hungarian method for the assignment problem", Naval Research Logistics Quarterly 2(1-2), 1955.
- D. F. Crouse, "On implementing 2D rectangular assignment algorithms", IEEE Trans. Aerospace and Electronic Systems 52(4), 2016.
- R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press, 2004.
- D. M. W. Powers, "Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness and Correlation", Journal of Machine Learning Technologies 2(1), 2011.

Proyek ini dikembangkan dari `https://github.com/antoinekeller/3d_bbox_highway.git`.

Pengembang: Angelino Zuliano Hutapea ([@angelinozulianohutapea](https://github.com/angelinozulianohutapea))
