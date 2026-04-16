# Tutorial Dashboard Streamlit + YOLO + Label Studio

Tutorial ini menjelaskan cara menjalankan dashboard yang bisa:

1. capture otomatis dari kamera
2. menjalankan inference YOLO
3. mengirim gambar hasil capture ke Label Studio
4. mengirim pre-label hasil deteksi YOLO ke Label Studio

## 1. Persiapan

Pastikan Python 3.10+ sudah tersedia.

Install dependency:

```bash
pip install -r requirements-dev.txt
```

Kalau kamu punya model YOLO sendiri, siapkan file `.pt` seperti `best.pt`.

## 2. Konfigurasi environment

Salin file `.env.example` menjadi `.env`, lalu isi sesuai kebutuhan:

```powershell
Copy-Item .env.example .env
```

```env
YOLO_MODEL_PATH=yolo11n.pt
CAPTURE_DIR=data/captures
EXPORT_DIR=data/exports
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=isi_api_key_label_studio
LABEL_STUDIO_PROJECT_TITLE=YOLO Camera Dashboard
LABEL_STUDIO_LOCAL_ROOT=data
```

Catatan:

- `YOLO_MODEL_PATH` bisa diarahkan ke model custom, misalnya `models/best.pt`
- `LABEL_STUDIO_LOCAL_ROOT` harus menjadi parent folder dari image capture
- default project ini menyimpan image di `data/captures`, jadi root `data` sudah cocok

## 3. Menjalankan dashboard

Jalankan perintah berikut:

```bash
streamlit run app.py
```

Setelah browser terbuka:

- isi `Path model YOLO`
- atur confidence dan IoU threshold
- jika perlu, isi filter label seperti `person, helmet`
- aktifkan `Auto capture kamera`

## 4. Menjalankan live detection

Masuk ke tab `Live Camera`.

Langkahnya:

1. klik `START` pada komponen kamera
2. izinkan browser mengakses webcam
3. hasil deteksi akan tampil di video stream
4. frame terakhir yang diproses juga tampil di dashboard
5. capture otomatis akan masuk ke folder `data/captures/`

Kalau mau simpan manual:

- klik `Simpan snapshot sekarang`

## 5. Inference dari image upload

Masuk ke tab `Image Inference`.

Di tab ini kamu bisa:

- upload file JPG, JPEG, atau PNG
- lihat bounding box dan confidence
- simpan image tersebut ke folder capture agar bisa ikut dikirim ke Label Studio

Ini berguna kalau kamu ingin uji model tanpa membuka live camera.

## 6. Menjalankan Label Studio

Supaya Label Studio bisa membaca hasil capture, jalankan dengan local files serving aktif.

### Opsi Docker

```powershell
docker run -it -p 8080:8080 -v ${PWD}/data:/label-studio/files --env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true --env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files heartexlabs/label-studio:latest label-studio
```

Penjelasan:

- host folder `data` di-mount ke `/label-studio/files`
- Label Studio akan membaca image lewat URL `/data/local-files/?d=...`
- karena image disimpan di `data/captures`, maka path relatifnya valid

### Ambil API Key Label Studio

1. buka Label Studio
2. login
3. masuk ke profile atau account settings
4. copy API token
5. simpan di `.env` atau isi langsung di tab `Label Studio`

## 7. Sync capture ke Label Studio

Masuk ke tab `Label Studio`.

Isi field berikut:

- `Label Studio URL`
- `Label Studio API Key`
- `Nama project Label Studio`
- `Document root Label Studio`

Lalu klik `Sync ke Label Studio`.

Yang terjadi setelah klik sync:

1. dashboard cek project Label Studio berdasarkan nama
2. kalau belum ada, project baru dibuat otomatis
3. label config dibuat dari class model YOLO
4. setiap image di `data/captures/` diubah jadi task Label Studio
5. prediksi YOLO ikut dikirim sebagai pre-label
6. file manifest JSON juga disimpan ke `data/exports/`

## 8. Melabeli data di Label Studio

Setelah task masuk:

1. buka project di Label Studio
2. pilih task
3. review pre-label dari YOLO
4. koreksi bounding box jika perlu
5. simpan annotation

Dengan alur ini kamu bisa mempercepat labeling karena box awal sudah dihasilkan model.

## 9. Export hasil labeling untuk YOLO training

Sesudah labeling selesai:

1. buka project di Label Studio
2. pilih menu export
3. pilih format `YOLO`
4. download hasil export

Hasil export itu bisa dipakai lagi untuk training model YOLO yang lebih bagus.

## 10. Workflow yang disarankan

Workflow paling enak biasanya seperti ini:

1. pakai tab `Live Camera` untuk kumpulkan data real
2. sync semua capture ke Label Studio
3. review dan koreksi pre-label
4. export dataset YOLO
5. train ulang model
6. ganti `YOLO_MODEL_PATH` ke model baru
7. ulangi proses untuk iterasi berikutnya

## 11. Troubleshooting

### Kamera tidak muncul

- pastikan browser diizinkan mengakses webcam
- coba reload halaman Streamlit
- pastikan tidak ada aplikasi lain yang sedang mengunci kamera

### Model YOLO gagal dibuka

- cek path model di sidebar
- kalau pakai model custom, pastikan file `.pt` benar
- kalau pakai model bawaan seperti `yolo11n.pt`, koneksi internet mungkin dibutuhkan saat download pertama

### Task masuk ke Label Studio tapi image tidak tampil

- biasanya `LABEL_STUDIO_LOCAL_ROOT` tidak cocok dengan folder yang di-mount ke container
- pastikan `data/` di host memang di-mount ke `/label-studio/files`
- pastikan `Document root Label Studio` di dashboard menunjuk ke folder yang sama

### Task dobel masuk ke Label Studio

- dashboard menyimpan index lokal di `data/exports/label_studio_sync_index.json`
- kalau ingin kirim ulang semua image, centang `Import ulang semua capture`

## 12. Menjalankan test

Jalankan:

```bash
pytest
```

Test yang ada saat ini fokus ke:

- pembentukan task Label Studio
- konversi path local files
- penyimpanan capture dan metadata

## 13. File penting

- `app.py` untuk dashboard utama
- `src/yolo_dashboard/yolo_inference.py` untuk loading model dan deteksi
- `src/yolo_dashboard/webrtc_processor.py` untuk live camera callback
- `src/yolo_dashboard/label_studio.py` untuk integrasi API Label Studio
- `src/yolo_dashboard/storage.py` untuk simpan capture dan metadata

## 14. Pengembangan lanjut

Kalau mau dilanjutkan, fitur berikut cocok ditambah:

- export YOLO langsung dari dashboard
- training trigger dari dashboard
- filtering berdasarkan confidence per class
- batch import video
- statistik dataset dan annotation progress
