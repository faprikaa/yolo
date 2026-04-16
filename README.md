# YOLO Camera Dashboard

Dashboard Streamlit ini sekarang mendukung alur end-to-end berikut:

- capture dari kamera browser
- inference YOLO dari base model atau model hasil training
- sync image ke Label Studio memakai Python SDK resmi `label-studio-sdk`
- menjalankan Label Studio dari instalasi `pip install label-studio`
- export anotasi Label Studio ke format YOLO langsung dari Streamlit
- menyiapkan `data.yaml` dan training ulang model Ultralytics langsung dari Streamlit

## Fitur utama

- Live camera detection memakai `streamlit-webrtc`
- Auto save frame hasil kamera ke `data/captures/`
- Inference dari image upload
- Integrasi Label Studio via SDK resmi, bukan REST manual
- Workflow Label Studio berbasis `pip`, bukan bergantung Docker
- Menu training untuk export dataset YOLO, split train/val, dan start training
- Selector model untuk base model YOLO dan model hasil training sendiri

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
5. Buka tab `Training` untuk export YOLO dari Label Studio, siapkan dataset, lalu training model baru.
6. Setelah training selesai, pilih `best.pt` hasil training sebagai model aktif dari sidebar atau tombol cepat di tab `Training`.

Tutorial lengkap ada di [TUTORIAL.md](TUTORIAL.md).
