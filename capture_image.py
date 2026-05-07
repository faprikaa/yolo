import cv2
import os
import time

os.makedirs("dataset/images", exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Kamera tidak terdeteksi!")
    exit()

print("Kamera OK! Warming up...")
time.sleep(2)  # tunggu kamera siap

# Buang beberapa frame pertama
for _ in range(10):
    cap.read()

print("Tekan 'S' untuk simpan frame, 'Q' untuk keluar")

count = 0

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Gagal baca frame, skip...")
        continue

    cv2.imshow("Capture", frame)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('s'):
        filename = f"dataset/images/frame_{count:04d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total: {count} gambar tersimpan")