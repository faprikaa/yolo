# YOLO Camera Dashboard

Dashboard ini dibuat dengan Streamlit untuk:

- auto capture dari kamera browser/device
- inference dan detection langsung memakai model YOLO
- kirim hasil capture ke Label Studio untuk proses labeling
- kirim pre-label dari hasil inferensi YOLO ke Label Studio

## Fitur utama

- Live camera detection memakai `streamlit-webrtc`
- Auto save frame hasil kamera ke `data/captures/`
- Inference dari image upload
- Integrasi Label Studio via REST API
- Manifest task JSON otomatis disimpan ke `data/exports/`
- Index sinkron lokal untuk mencegah import task dobel

## Struktur project

```text
app.py
src/yolo_dashboard/
tests/
data/captures/
data/exports/
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

4. Jalankan Streamlit:

```bash
streamlit run app.py
```

5. Jalankan Label Studio bila ingin labeling:

```powershell
docker run -it -p 8080:8080 -v ${PWD}/data:/label-studio/files --env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true --env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files heartexlabs/label-studio:latest label-studio
```

## Cara kerja singkat

1. Buka tab `Live Camera` lalu klik `START`.
2. Aktifkan `Auto capture kamera` jika ingin frame tersimpan otomatis.
3. Buka tab `Label Studio` untuk sync capture ke project labeling.
4. Setelah labeling selesai di Label Studio, export dataset dari UI Label Studio ke format YOLO untuk training lanjutan.

Tutorial lengkap ada di [TUTORIAL.md](TUTORIAL.md).
