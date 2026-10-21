# 3D Bounding Box Tracking dan Penghitungan Mobil di Jalan

Pelacakan dan penghitungan kendaraan dengan kotak 3D pada video jalan raya dari satu kamera statis (misalnya di jembatan penyeberangan). Detektor CenterNet menghasilkan titik pusat dan ukuran kendaraan, posisinya diproyeksikan ke bidang jalan, dilacak dengan Kalman Filter, lalu dihitung saat melewati garis virtual. Hasilnya digambar sebagai kotak 3D lengkap dengan ID, kelas kendaraan (car, van, truck), dan estimasi kecepatan.

## Fitur

- Deteksi kendaraan dengan model CenterNet (PyTorch, `.pth`, backbone ResNet-18, input 1280x720)
- Ekstraksi puncak heatmap dengan max-pooling 3x3 (`--nms pool`, mengikuti CenterNet asli) atau cara lama (`--nms legacy`)
- Proyeksi titik kontak ban ke bidang jalan memakai kalibrasi kamera (pitch, tinggi, focal length)
- Kalman Filter dengan noise pengukuran yang bergantung pada jarak dan dimensi kendaraan yang terkunci, sehingga kotak lebih stabil
- Asosiasi deteksi ke track memakai algoritma Hungarian (`scipy.optimize.linear_sum_assignment`) dengan gate adaptif (objek jauh lebih longgar)
- Asosiasi dua tahap ala ByteTrack (opsional, `--conf-low`): deteksi skor rendah hanya menyambung track yang sudah ada, tidak membuat track baru
- Objek baru baru digambar setelah beberapa kali terdeteksi, supaya deteksi palsu tidak muncul
- Penghitungan kendaraan yang melewati garis virtual, per kelas (car, van, truck), dengan log CSV
- Cache deteksi mentah (`--cache`), sehingga detektor cukup dijalankan sekali dan eksperimen tracking berikutnya jauh lebih cepat
- Mode debug yang menandai setiap deteksi mentah (lolos atau ditolak beserta alasannya)
- Alat penyetel kalibrasi interaktif dengan slider (`calibrate_tuner.py`)
- Alat pelabelan titik mobil dan fine-tuning head heatmap, plus skrip evaluasi dan analisis

## Struktur Proyek

### Program utama

| File                       | Fungsi                                                                                  |
| -------------------------- | --------------------------------------------------------------------------------------- |
| `multi_object_tracking.py` | Program utama: deteksi, tracking, penghitungan, tampilan, dan log hasil                 |
| `kalman_filter.py`         | Kalman Filter (state: x, y, vy, w, h, l)                                                |
| `object_detector.py`       | Pemuatan model CenterNet, inferensi, dan ekstraksi puncak heatmap                       |
| `vehicle_shape.py`         | Bentuk kendaraan adaptif: kelas, dimensi, yaw, dan gambar kotak berorientasi            |
| `utils.py`                 | Proyeksi 3D ke 2D, ray tracing ke bidang jalan, dan fungsi menggambar                   |
| `calibrate_tuner.py`       | Alat penyetel kalibrasi kamera dengan slider                                            |

### Alur pelabelan dan fine-tuning

| File                   | Fungsi                                                                                         |
| ---------------------- | ---------------------------------------------------------------------------------------------- |
| `label_points.py`      | Alat pelabelan: satu klik per mobil, disimpan ke `labels.json`                                 |
| `finetune_heatmap.py`  | Fine-tuning head heatmap CenterNet dengan label titik, menghasilkan `centernet-ft.pth`         |
| `cek_fp.py`            | Evaluasi benar/palsu/terlewat per frame dan penyimpanan gambar frame dengan deteksi palsu terbanyak |

### Skrip analisis hasil

| File                  | Fungsi                                                                                         |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| `lembar_hitungan.py`  | Membuat lembar gambar (`lembar_N.png`), satu frame per hitungan, untuk memeriksa hasil counting |
| `analisis_cache.py`   | Melihat skor deteksi per frame dari cache pada area tertentu (konfigurasi ada di dalam file)   |
| `bandingkan_ukuran.py`| Membandingkan panjang terukur deteksi fine-tune dengan model asli di lokasi yang sama          |

## Instalasi

Butuh Python 3.10 atau lebih baru (diuji di Python 3.12, Ubuntu).

```
git clone https://github.com/angelinozulianohutapea/3d-car-classification-counting.git
cd 3d-car-classification-counting
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy opencv-python torch torchvision tqdm
```

Setelah `source venv/bin/activate`, perintah `python` di bawah ini merujuk ke Python milik venv. Di luar venv, pakai `python3`.

## Menyiapkan Model dan Video

File model (`.pth`) dan video tidak disertakan di repositori karena ukurannya besar.

- Model CenterNet: `<isi tautan unduhan atau cara melatih>`
- Video contoh: `<isi sumber video>`

Taruh keduanya di folder proyek, atau gunakan path lengkapnya saat menjalankan program.

**Ukuran video harus 1280x720.** Ukuran input detektor (`ObjectDetector.INPUT_WIDTH/HEIGHT`), titik pusat kamera pada matriks `K` (640, 360), dan tinggi gambar 720 pada proyeksi ke jalan semuanya tertulis tetap di kode. Video dengan resolusi lain perlu diubah ukurannya (atau kodenya disesuaikan) sebelum dipakai.

## Cara Pakai

### 1. Kalibrasi kamera

Nilai kalibrasi bawaan (pitch 9.8 derajat, tinggi kamera 8.30 m, focal 770 px) berasal dari simulator CARLA, jadi untuk video nyata perlu disetel ulang.

```
python calibrate_tuner.py VIDEO.mp4 --frame 100
```

Opsi: `--frame` (nomor frame), `--pitch` (awal, derajat), `--height` (awal, meter), `--focal` (awal, piksel).

Urutan menyetel:

1. Setel `PITCH` sampai garis horizon kuning pas dengan horizon asli.
2. Setel `FOCAL` dan `HEIGHT` sampai garis grid mengarah ke titik hilang yang sama dengan marka jalan.
3. Geser `GRID_OX` supaya garis grid menempel di marka lajur.
4. Geser `REF_X` dan `REF_Z` sampai kotak hijau berada di atas satu mobil, dengan alas menempel di aspal.
5. Cek di jarak dekat dan jauh, lalu tekan `q` (ESC untuk batal).

Terminal mencetak tiga baris nilai. Salin ke `multi_object_tracking.py` (`pitch_matrix`, tinggi kamera di `to_world_from_camera`, dan `K`). Slider `GRID_OX`, `GRID_OZ`, `LANE_W`, `REF_X`, dan `REF_Z` hanya alat bantu visual dan tidak ikut dicetak.

Untuk percobaan cepat, tinggi kamera bisa ditimpa tanpa mengubah kode lewat `--cam-height`.

### 2. Menjalankan tracking dan penghitungan

```
python multi_object_tracking.py VIDEO.mp4 PATH_MODEL.pth
```

Contoh tanpa jendela (misalnya lewat SSH) dan tanpa menyimpan PNG tiap frame:

```
python multi_object_tracking.py VIDEO.mp4 PATH_MODEL.pth --no-show --no-save --line-y 15
```

Contoh dengan cache deteksi dan asosiasi dua tahap:

```
python multi_object_tracking.py VIDEO.mp4 PATH_MODEL.pth --cache det_low.pkl --conf 0.5 --conf-low 0.3 --no-show --no-save
```

Pada jalankan pertama, deteksi mentah dibuat dan disimpan ke `det_low.pkl`. Pada jalankan berikutnya file itu dimuat dan detektor tidak dijalankan lagi.

#### Opsi

| Opsi              | Keterangan                                                                                                      |
| ----------------- | --------------------------------------------------------------------------------------------------------------- |
| `--conf`          | Ambang kepercayaan deteksi (default 0.5)                                                                        |
| `--conf-low`      | Aktifkan asosiasi dua tahap (ByteTrack). Deteksi skor antara `--conf-low` dan `--conf` hanya menyambung track yang ada. Harus > 0 dan lebih kecil dari `--conf` |
| `--line-y`        | Posisi garis virtual counting, dalam meter dari kamera (default 15.0)                                           |
| `--nms`           | `pool` (default, max-pool 3x3) atau `legacy` (cara lama)                                                        |
| `--cache`         | File cache deteksi mentah. Jika ada, dimuat; jika belum ada, dibuat. Hanya untuk `--nms pool`                   |
| `--cache-conf`    | Ambang skor terendah saat membuat cache (default 0.2)                                                           |
| `--max-width`     | Batas atas lebar kendaraan dalam meter; 0 = nonaktif (default 2.5)                                              |
| `--conflict-dist` | Dua track yang lebih dekat dari jarak ini (meter) dianggap duplikat (default 3.0)                               |
| `--cam-height`    | Timpa tinggi kamera (meter) untuk percobaan kalibrasi (default dari kode, 8.30)                                 |
| `--no-show`       | Tanpa jendela (untuk SSH atau VS Code remote)                                                                   |
| `--no-save`       | Jangan simpan PNG tiap frame (jauh lebih cepat)                                                                 |
| `--play`          | Langsung jalan tanpa menunggu tombol (mode jendela)                                                             |
| `-f N`            | Putar otomatis sampai frame ke-N, setelah itu berhenti di tiap frame                                            |
| `--debug`         | Tandai semua deteksi mentah: hijau = lolos filter, kuning = lolos tetapi skor rendah, merah = ditolak (dengan alasan) |

Tombol saat jendela tampil: `SPACE` untuk play/pause, `ESC` untuk keluar. Tanpa `--play` atau `-f`, program berhenti di tiap frame dan menunggu tombol.

#### Cara kerja penghitungan

- Garis virtual digambar merah pada kedalaman `--line-y`, selebar jalur valid (|x| ≤ 11 m).
- Penghitungan tidak membedakan arah: track dianggap melintas saat posisinya berpindah dari satu sisi garis ke sisi lainnya.
- Setiap track dihitung paling banyak sekali, dan hanya setelah track itu layak tampil (terdeteksi minimal 3 kali), supaya deteksi palsu sesaat tidak masuk hitungan.
- Kelas yang dihitung: `car`, `van`, `truck`.

#### Filter deteksi

Sebelum masuk ke tracking, deteksi mentah dibuang bila:

| Alasan (label `--debug`) | Kondisi                                                |
| ------------------------ | ------------------------------------------------------ |
| `luar jalur`             | posisi lateral di jalan \|x\| > 11 m                   |
| `w kecil`                | lebar < 0.6 m                                          |
| `tinggi`                 | tinggi < 1 m atau > 5 m                                |
| `l pendek`               | panjang < 3 m                                          |
| `h>l`                    | tinggi lebih besar dari panjang                        |

Ringkasan jumlah deteksi yang dibuang per alasan dicetak di akhir program.

#### Keluaran

- `counting_log.csv`: satu baris per kendaraan yang dihitung.

  | Kolom                                    | Isi                                                      |
  | ---------------------------------------- | -------------------------------------------------------- |
  | `frame`                                  | frame saat kendaraan dihitung                            |
  | `id`                                     | ID tampilan                                              |
  | `kelas`                                  | kelas hasil voting                                       |
  | `panjang_m`, `lebar_m`, `tinggi_m`       | dimensi terukur (median)                                 |
  | `x_m`                                    | posisi lateral saat melintas                             |
  | `suara_car`, `suara_van`, `suara_truck`  | jumlah suara tiap kelas                                  |

- Ringkasan di terminal: hitungan per kelas dan total, daftar track yang tampil tetapi tidak dihitung (beserta keterangan), detail tiap kendaraan yang dihitung, dan statistik dimensi deteksi yang lolos (min / median / maks) dibandingkan dengan acuan mobil nyata.
- `video_tracking_carla/image_XXXX.png`: frame hasil tiap frame (kecuali dengan `--no-save`).
- File cache (`--cache`): berformat pickle. Muat hanya file yang Anda buat sendiri.

## Melabeli dan Fine-tuning

Model CenterNet awal dilatih pada data simulasi CARLA. Untuk menyesuaikannya dengan video nyata, head heatmap (`outc`) dapat dilatih ulang memakai label titik. Backbone serta head offset dan ukuran dibekukan, dan tidak ada augmentasi.

### 1. Melabeli titik mobil

```
python label_points.py VIDEO.mp4 --out labels.json --cache det_low.pkl
```

| Kontrol        | Fungsi                                                          |
| -------------- | --------------------------------------------------------------- |
| Klik kiri      | tambah titik (titik kontak mobil dengan jalan)                  |
| Klik kanan     | hapus titik terdekat                                            |
| `d` / `a`      | frame +1 / −1                                                   |
| `w` / `s`      | frame +5 / −5                                                   |
| `u`            | batalkan titik terakhir                                         |
| `SPASI`        | tandai frame SELESAI (boleh 0 titik sebagai contoh negatif)     |
| `m`            | tampil/sembunyi titik deteksi model dari cache (kuning)         |
| `q` / `ESC`    | simpan dan keluar                                               |

Opsi lain: `--start` (frame awal), `--scale` (perkecil jendela bila layar kecil). Pada frame yang ditandai SELESAI, **semua** mobil harus diberi titik, karena mobil tanpa titik dianggap latar saat pelatihan.

### 2. Fine-tuning

```
python finetune_heatmap.py VIDEO.mp4 centernet-3d-bbox.pth --val-from 330
```

| Opsi         | Keterangan                                                                  |
| ------------ | --------------------------------------------------------------------------- |
| `--labels`   | file label (default `labels.json`)                                          |
| `--out`      | model keluaran (default `centernet-ft.pth`)                                 |
| `--val-from` | frame dengan nomor ≥ nilai ini dipakai sebagai validasi                     |
| `--epochs`   | jumlah epoch (default 60)                                                   |
| `--lr`       | learning rate Adam (default 1e-4)                                           |
| `--sigma`    | lebar Gaussian target heatmap, satuan sel (default 0.8)                     |

Fungsi kerugian memakai focal loss untuk heatmap seperti pada CenterNet (Zhou, Wang, Krähenbuhl, 2019). Sebelum dan sesudah pelatihan dicetak jumlah deteksi benar, palsu, dan terlewat pada conf 0.2, 0.3, dan 0.5 (deteksi dianggap benar bila berjarak ≤ 48 piksel dari titik label).

### 3. Evaluasi dan analisis

```
python cek_fp.py VIDEO.mp4 centernet-ft.pth --conf 0.3 --from 330 --to 472 --out cek_ft
python lembar_hitungan.py VIDEO.mp4 --log counting_log.csv --line-y 15
```

`cek_fp.py` mencetak jumlah benar, palsu, dan terlewat, lalu menyimpan gambar frame dengan deteksi palsu terbanyak (lingkaran merah = label, hijau = deteksi benar, kuning = deteksi palsu). `lembar_hitungan.py` membuat lembar gambar berisi satu frame per kendaraan yang dihitung, dengan penanda posisi saat melewati garis.

`analisis_cache.py` dan `bandingkan_ukuran.py` adalah skrip eksperimen: nama file cache dan rentang frame ditulis langsung di dalam file, jadi sesuaikan sebelum dipakai.

## Menyetel Hasil

Parameter utama ada di bagian konfigurasi `multi_object_tracking.py`, di `kalman_filter.py`, dan sebagai opsi baris perintah.

| Gejala                                                       | Yang disesuaikan                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------------ |
| Kotak masih goyang                                           | Naikkan nilai di `measurement_noise()` (`kalman_filter.py`)        |
| Kotak tertinggal saat mobil mengerem atau mempercepat        | Naikkan `MAX_ACCEL`                                                |
| Ukuran kotak berubah-ubah                                    | Turunkan `DIM_STD_PER_STEP`                                        |
| Objek terlambat muncul                                       | Turunkan `MIN_HITS_TO_DISPLAY` (default 3)                         |
| ID sering berganti untuk mobil yang sama                     | Naikkan `MAX_LOST_FRAMES` (default 30) atau longgarkan gate di `_match()` |
| Mobil terdeteksi hilang-timbul sehingga tidak terhitung      | Aktifkan `--conf-low` dan/atau turunkan `--conf`                   |
| Satu mobil menjadi dua track                                 | Naikkan `--conflict-dist`                                          |
| Deteksi palsu masuk hitungan                                 | Naikkan `--conf`, atau periksa dengan `--debug` alasan penolakannya |
| Mobil lebar terpotong ukurannya                              | Ubah atau nonaktifkan `--max-width`                                |

## Keterbatasan

- Model CenterNet awal dilatih pada data simulasi CARLA, sehingga ada domain gap saat dipakai pada video jalan nyata. Deteksi dan ukuran kotak bisa kurang akurat, termasuk mobil yang tidak terdeteksi dan deteksi palsu.
- Fine-tuning hanya melatih head heatmap (posisi pusat) dari label titik. Estimasi ukuran (lebar, tinggi, panjang) masih berasal dari model asli, sehingga klasifikasi car/van/truck yang bergantung pada ukuran masih bisa keliru, misalnya sedan terbaca sebagai truck.
- Data latih dan validasi fine-tuning berasal dari satu video yang sama (dibagi menurut nomor frame), jadi hasilnya belum menunjukkan kemampuan pada video lain.
- Kalibrasi kamera disetel manual dengan mata. Pitch, tinggi, dan focal length saling memengaruhi, jadi beberapa kombinasi bisa tampak cocok.
- Bidang jalan diasumsikan datar dan kamera statis. Video harus berukuran 1280x720 (lihat bagian Menyiapkan Model dan Video).
- Penghitungan tidak membedakan arah, dan track yang tidak sempat layak tampil sebelum melewati garis tidak dihitung.
- Kecepatan adalah estimasi dari hasil kalibrasi, bukan pengukuran yang sudah divalidasi.

## Kredit dan Lisensi

Proyek ini dikembangkan dari `https://github.com/antoinekeller/3d_bbox_highway.git`.

Rujukan metode:

- X. Zhou, D. Wang, P. Krähenbühl, "Objects as Points", arXiv:1904.07850, 2019 (CenterNet, ekstraksi puncak heatmap dan focal loss)
- Y. Zhang dkk., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", arXiv:2110.06864, 2022 (asosiasi dua tahap)

Pengembang: Angelino Zuliano Hutapea ([@angelinozulianohutapea](https://github.com/angelinozulianohutapea))