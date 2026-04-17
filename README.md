# YOLO Camera Dashboard

Dashboard Streamlit ini sekarang mendukung alur end-to-end berikut:

- capture dari kamera browser
- inference YOLO dari base model atau model hasil training
- sync image ke Label Studio memakai Python SDK resmi `label-studio-sdk`
- menjalankan Label Studio dari instalasi `pip install label-studio`
- upload ZIP hasil export YOLO dari Label Studio ke Streamlit
- export anotasi Label Studio ke format YOLO langsung dari Streamlit
- memakai folder dataset YOLO lokal yang sudah ada dan valid
- menyiapkan `data.yaml` dan training ulang model Ultralytics langsung dari Streamlit

## Fitur utama

- Live camera detection memakai `streamlit-webrtc`
- Auto save frame hasil kamera ke `data/captures/`
- Inference dari image upload
- Integrasi Label Studio via SDK resmi, bukan REST manual
- Workflow Label Studio berbasis `pip`, bukan bergantung Docker
- Menu training untuk upload/export dataset YOLO, pilih folder dataset lokal, dan start training
- Selector model untuk base model YOLO11/YOLOv8 dan model hasil training sendiri
- Tab training dengan parameter training, pilihan device `GPU 0` / `GPU 1` / `CPU`, progress epoch, dan resource usage

## Model yang tersedia

Di sidebar, aplikasi bisa memakai tiga sumber model:

- `Base YOLO` untuk model bawaan Ultralytics
- `Model hasil training` untuk file `.pt` yang muncul dari hasil training project ini
- `Path custom` untuk model `.pt` lain yang ingin dipakai manual

Model yang tersedia di opsi `Base YOLO` saat ini:

- `yolo11n.pt`
- `yolo11s.pt`
- `yolo11m.pt`
- `yolo11l.pt`
- `yolo11x.pt`
- `yolov8n.pt`
- `yolov8s.pt`
- `yolov8m.pt`
- `yolov8l.pt`
- `yolov8x.pt`

Arti suffix model:

- `n` = nano
  Model paling ringan dan paling cepat. Cocok untuk live cam inference di laptop biasa, eksperimen cepat, atau perangkat dengan resource terbatas.
- `s` = small
  Sedikit lebih berat dari `n`, tapi biasanya akurasi lebih stabil. Cocok untuk penggunaan harian kalau tetap butuh inference cepat.
- `m` = medium
  Titik tengah antara kecepatan dan akurasi. Cocok untuk training dan inference saat resource GPU sudah lebih memadai.
- `l` = large
  Lebih berat, biasanya dipakai saat fokus ke akurasi dibanding latency. Cocok untuk training di GPU yang lebih kuat.
- `x` = extra large
  Model paling besar di daftar ini. Cocok untuk eksperimen akurasi maksimum, bukan untuk device yang terbatas.

Fungsi model di dalam aplikasi ini:

- Model aktif dipakai untuk `Live Cam Inference`
- Model aktif dipakai untuk `Image Inference`
- Model aktif dipakai untuk pre-label saat sync ke Label Studio
- Model aktif juga dipakai sebagai starting weights saat training dari tab `Training`

Panduan memilih model:

- Pakai `yolo11n.pt` atau `yolov8n.pt` kalau targetnya webcam real-time dan hardware terbatas
- Pakai `yolo11s.pt` atau `yolov8s.pt` kalau ingin sedikit naik akurasi tanpa terlalu berat
- Pakai `m`, `l`, atau `x` kalau fokus utama ada di kualitas deteksi dan training dilakukan di GPU
- Pakai `Model hasil training` kalau Anda sudah punya `best.pt` dari dataset sendiri dan ingin inference yang lebih sesuai domain data Anda
- Pakai `Path custom` kalau file model berada di luar folder hasil training aplikasi ini

## Parameter Training

Tab `Training` sekarang mendukung tiga sumber dataset:

- `Upload ZIP` untuk file export YOLO dari Label Studio
- `Export via API` untuk mengambil export langsung dari project Label Studio
- `Folder Dataset` untuk dataset YOLO lokal yang sudah ada dan valid

Dataset lokal dianggap valid jika:

- path folder atau file `data.yaml` memang ada
- `data.yaml` punya entry `train`, `val`, dan `names`
- path train/val berisi image yang bisa dipakai
- struktur label YOLO terdeteksi

Parameter training yang bisa diatur dari UI:

- `Nama run training`
  Nama folder hasil training di `data/runs/`.
- `Epochs`
  Jumlah iterasi training penuh terhadap dataset.
- `Batch size`
  Jumlah image yang diproses per step training.
- `Image size`
  Ukuran image input saat training. Semakin besar biasanya detail lebih baik, tapi beban GPU/RAM naik.
- `Patience`
  Jumlah epoch tanpa peningkatan sebelum early stopping menghentikan training lebih cepat.
- `Device training`
  Pilihan device yang dipakai training seperti `Auto`, `CPU`, `GPU 0`, atau `GPU 1`.
- `Workers`
  Jumlah worker data loader. Biasanya makin tinggi bisa mempercepat loading data, tapi memakai resource CPU lebih banyak.
- `Optimizer`
  Optimizer yang dipakai Ultralytics, misalnya `auto`, `SGD`, `Adam`, atau `AdamW`.
- `Learning rate`
  Nilai awal learning rate (`lr0`) untuk training.

Selama training berjalan, dashboard menampilkan:

- progress epoch
- metric terakhir dari `results.csv`
- resource usage CPU
- resource usage RAM
- resource usage GPU

## Penjelasan Slider UI

Beberapa pengaturan di dashboard memakai slider karena nilainya biasanya perlu dicari titik yang pas sesuai data dan hardware.

Slider di sidebar inference:

- `Confidence threshold`
  Menentukan skor minimum agar deteksi dianggap valid.
  Nilai lebih tinggi berarti model lebih ketat dan jumlah box biasanya berkurang.
  Nilai lebih rendah berarti model lebih permisif dan lebih banyak box bisa muncul, termasuk false positive.
- `IoU threshold`
  Dipakai saat proses Non-Max Suppression untuk menentukan kapan dua box dianggap saling overlap terlalu besar.
  Nilai lebih rendah membuat box duplikat lebih cepat dibuang.
  Nilai lebih tinggi membuat model lebih toleran terhadap box yang saling tumpang tindih.
- `Inference image size`
  Ukuran image yang dipakai saat inference.
  Nilai lebih besar bisa membantu objek kecil lebih terlihat, tetapi inference jadi lebih berat dan lebih lambat.
  Nilai lebih kecil membuat inference lebih ringan dan lebih cepat.

Slider di tab training:

- `Proporsi train split`
  Menentukan berapa persen data masuk ke train set saat dataset disiapkan dari ZIP export Label Studio.
  Contoh `0.80` berarti sekitar 80% data dipakai untuk training dan 20% sisanya untuk validation.
  Nilai ini dipakai saat sumber dataset berasal dari `Upload ZIP` atau `Export via API`.

Panduan cepat:

- Naikkan `Confidence threshold` kalau terlalu banyak deteksi salah
- Turunkan `Confidence threshold` kalau objek benar sering tidak tertangkap
- Turunkan `IoU threshold` kalau box ganda terlalu sering muncul
- Naikkan `Inference image size` kalau objek kecil sulit terbaca dan hardware masih kuat
- Pakai `train split` sekitar `0.80` sampai `0.90` untuk kebanyakan eksperimen awal

## Struktur project

```text
app.py
src/yolo_dashboard/
tests/
data/captures/
data/exports/
data/datasets/
data/runs/
README.md
TUTORIAL.md
```

## Quick start

1. Buat virtual environment.
2. Install dependency:

```bash
pip install -r requirements-dev.txt
```

3. Salin environment file:

```powershell
Copy-Item .env.example .env
```

4. Jalankan Label Studio dari package `pip`:

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD\data"
label-studio start
```

5. Jalankan Streamlit:

```bash
streamlit run app.py
```

## Cara kerja singkat

1. Pilih model aktif di sidebar: base model YOLO, model hasil training, atau path custom.
2. Buka tab `Live Camera` untuk capture data.
3. Buka tab `Label Studio` untuk sync capture ke project labeling.
4. Review dan simpan anotasi di Label Studio.
5. Buka tab `Training` untuk upload ZIP export Label Studio, export via API, atau pilih folder dataset lokal, lalu training model baru.
6. Setelah training selesai, pilih `best.pt` hasil training sebagai model aktif dari sidebar atau tombol cepat di tab `Training`.

Tutorial lengkap ada di [TUTORIAL.md](TUTORIAL.md).
