# Tutorial Dashboard Streamlit + YOLO + Label Studio

Tutorial ini menjelaskan cara menjalankan dashboard yang sekarang bisa:

1. capture otomatis dari kamera
2. menjalankan inference YOLO
3. sync image ke Label Studio memakai SDK resmi
4. export dataset YOLO langsung dari Streamlit
5. training ulang model YOLO langsung dari Streamlit
6. memilih model base YOLO atau model hasil training sendiri
7. memakai folder dataset YOLO lokal yang sudah ada dan valid

## 1. Persiapan

Pastikan Python 3.10+ sudah tersedia.

Install dependency:

```bash
pip install -r requirements-dev.txt
```

Dependency penting untuk workflow baru:

- `label-studio` untuk server labeling via pip
- `label-studio-sdk` untuk create project, import task, dan export dataset
- `ultralytics` untuk inference dan training YOLO

## 2. Konfigurasi environment

Salin `.env.example` menjadi `.env`.

```powershell
Copy-Item .env.example .env
```

Contoh isi `.env`:

```env
YOLO_MODEL_PATH=yolo11n.pt
CAPTURE_DIR=data/captures
EXPORT_DIR=data/exports
DATASET_DIR=data/datasets
YOLO_RUNS_DIR=data/runs
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=isi_api_key_label_studio
LABEL_STUDIO_PROJECT_TITLE=YOLO Camera Dashboard
LABEL_STUDIO_LOCAL_ROOT=data
```

Catatan:

- `YOLO_MODEL_PATH` adalah model default saat app pertama kali dibuka.
- `DATASET_DIR` dipakai untuk dataset hasil export Label Studio yang sudah disiapkan ke format Ultralytics.
- `YOLO_RUNS_DIR` dipakai untuk output training seperti `best.pt` dan `last.pt`.
- `LABEL_STUDIO_LOCAL_ROOT` harus menjadi parent folder dari image capture.

## 3. Menjalankan Label Studio via pip

Project ini tidak lagi mengandalkan Docker sebagai workflow utama.

Jalankan Label Studio seperti ini:

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD\data"
label-studio start
```

Penjelasan:

- `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` mengaktifkan local file serving
- `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` harus menunjuk ke root folder data project
- image capture ada di `data/captures`, jadi root `data` sudah benar

## 4. Jalankan Streamlit

```bash
streamlit run app.py
```

## 5. Pilih model YOLO

Di sidebar sekarang ada selector model:

- `Base YOLO` untuk model bawaan seperti `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`, `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`
- `Model hasil training` untuk memilih file `.pt` yang ada di `data/runs`
- `Path custom` kalau kamu punya model di lokasi lain

Model aktif ini dipakai untuk:

- live inference
- image inference
- pre-label ke Label Studio
- starting weights saat training dari tab `Training`

### Penjelasan slider di sidebar

- `Confidence threshold`
  Menentukan skor minimum supaya sebuah deteksi tetap ditampilkan.
  Kalau nilainya dinaikkan, deteksi akan lebih ketat.
  Kalau nilainya diturunkan, model akan lebih mudah menampilkan box, termasuk yang confidence-nya rendah.
- `IoU threshold`
  Dipakai saat penyaringan box yang saling overlap.
  Nilai lebih kecil biasanya membuang box duplikat lebih agresif.
  Nilai lebih besar membuat overlap lebih ditoleransi.
- `Inference image size`
  Ukuran image untuk proses inference.
  Nilai besar bisa membantu objek kecil lebih terbaca, tapi beban komputasi naik.
  Nilai kecil lebih cepat untuk live cam.

## 6. Live Camera

Masuk ke tab `Live Camera`.

Langkahnya:

1. klik `START` pada komponen kamera
2. izinkan browser mengakses webcam
3. aktifkan `Auto capture kamera` jika ingin frame tersimpan otomatis
4. review preview dan tabel deteksi

Semua capture masuk ke `data/captures/`.

## 7. Image Inference

Masuk ke tab `Image Inference`.

Di sini kamu bisa:

- upload JPG/JPEG/PNG
- lihat bounding box hasil inference
- simpan image ke folder capture agar bisa ikut masuk ke Label Studio

## 8. Sync ke Label Studio

Masuk ke tab `Label Studio`.

Isi field berikut:

- `Label Studio URL`
- `Label Studio API Key`
- `Nama project Label Studio`
- `Document root Label Studio`

Opsi tambahan:

- `Kirim prediksi YOLO sebagai pre-label`
- `Import ulang semua capture`

Saat tombol sync ditekan, dashboard akan:

1. mengambil class names dari model YOLO aktif
2. membuat project Label Studio kalau belum ada
3. mengubah capture menjadi task Label Studio
4. mengirim prediction sebagai pre-label jika diaktifkan
5. menyimpan manifest task di `data/exports/`

## 9. Melabeli data

Sesudah task masuk:

1. buka project di Label Studio
2. review bounding box hasil pre-label
3. koreksi bila perlu
4. simpan annotation

## 10. Export YOLO dan training langsung dari Streamlit

Masuk ke tab `Training`.

Workflow barunya:

Opsi sumber dataset:

- `Upload ZIP` kalau kamu sudah export YOLO manual dari UI Label Studio
- `Export via API` kalau ingin dashboard yang menarik export langsung dari project Label Studio
- `Folder Dataset` kalau kamu sudah punya dataset YOLO lokal dengan `data.yaml` yang valid

Slider di tab ini:

- `Proporsi train split`
  Menentukan pembagian data train dan validation saat dataset disiapkan dari export Label Studio.
  Contoh `0.80` berarti sekitar 80% data untuk training dan 20% untuk validation.

Contoh workflow `Export via API`:

1. isi `Label Studio URL` dan `API Key`
2. pilih project Label Studio
3. atur `Proporsi train split`
4. klik `Export YOLO dan siapkan dataset`
5. dashboard akan:
   - download archive export YOLO dari Label Studio
   - extract archive ke folder sementara
   - menyiapkan struktur `images/train`, `images/val`, `labels/train`, `labels/val`
   - membuat `data.yaml`
   - menyimpan dataset siap train ke `data/datasets/`
6. pilih dataset yang ingin dipakai
7. atur parameter training seperti `epochs`, `batch size`, `image size`, `patience`, `optimizer`, `learning rate`, `workers`, dan `device`
8. klik `Mulai training YOLO`

Kalau pakai `Folder Dataset`, masukkan salah satu dari:

- path ke folder dataset yang berisi `data.yaml`
- path langsung ke file `data.yaml`

Dataset lokal dianggap valid jika:

- `data.yaml` ada
- entry `train`, `val`, dan `names` ada
- path train/val berisi image
- label YOLO ikut terdeteksi

### Penjelasan parameter training

- `Nama run training`
  Nama folder hasil training di `data/runs`.
- `Epochs`
  Total epoch training.
- `Batch size`
  Jumlah image per batch.
- `Image size`
  Resolusi image input saat training.
- `Patience`
  Batas epoch tanpa improvement sebelum early stopping.
- `Device training`
  Device yang dipakai training seperti `Auto`, `CPU`, `GPU 0`, atau `GPU 1`.
- `Workers`
  Jumlah worker data loader.
- `Optimizer`
  Optimizer yang dipakai selama training.
- `Learning rate`
  Nilai awal learning rate.

### Panduan singkat mengatur slider

- Naikkan `Confidence threshold` kalau terlalu banyak false positive
- Turunkan `Confidence threshold` kalau objek sering lolos tidak terdeteksi
- Turunkan `IoU threshold` kalau box ganda terlalu sering muncul
- Naikkan `Inference image size` kalau objek kecil sulit terbaca dan hardware masih sanggup
- Pakai `Proporsi train split` sekitar `0.80` sampai `0.90` untuk percobaan awal

## 11. Hasil training

Output training masuk ke `data/runs/`.

File penting:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`

Setelah training selesai:

- model baru otomatis terdeteksi oleh selector sidebar pada rerun berikutnya
- kamu bisa klik tombol `Pakai best.pt sebagai model aktif`
- atau pilih sendiri dari opsi `Model hasil training`
- selama training berjalan, tab `Training` menampilkan progress epoch, metric terakhir, dan resource usage CPU/RAM/GPU

## 12. Struktur dataset hasil export

Dataset siap train yang dibangun dashboard akan berbentuk seperti ini:

```text
data/datasets/prepared_yolo_<timestamp>/
  data.yaml
  dataset_metadata.json
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Tujuan langkah ini adalah membuat export YOLO dari Label Studio langsung kompatibel untuk `YOLO(...).train(...)` di Ultralytics.

## 13. Menjalankan test

```bash
pytest
```

Test saat ini mencakup:

- pembentukan payload task Label Studio
- konversi local file URL
- penyimpanan capture dan metadata
- persiapan dataset YOLO dari ZIP export Label Studio
- discovery model hasil training

## 14. Troubleshooting

### Model base YOLO gagal dipakai

- cek koneksi internet untuk download weight bawaan saat pertama kali dipakai
- kalau offline, gunakan model `.pt` yang sudah ada lokal

### Capture berhasil sync tapi image tidak tampil di Label Studio

- pastikan `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` cocok dengan folder `data`
- pastikan field `Document root Label Studio` di tab `Label Studio` menunjuk ke folder yang sama

### Export berhasil tapi dataset tidak bisa ditrain

- pastikan project Label Studio sudah punya annotation yang tersimpan
- cek jumlah `Labeled images` di tab `Training`
- kalau nol, berarti export belum berisi label yang valid

### Model hasil training tidak muncul di sidebar

- pastikan training benar-benar menghasilkan file `.pt` di `data/runs`
- lakukan rerun Streamlit setelah training selesai

## 15. File penting

- `app.py` untuk dashboard Streamlit
- `src/yolo_dashboard/label_studio.py` untuk SDK Label Studio
- `src/yolo_dashboard/training.py` untuk export preparation dan training YOLO
- `src/yolo_dashboard/yolo_inference.py` untuk loading model dan inference
- `src/yolo_dashboard/storage.py` untuk simpan capture dan metadata
