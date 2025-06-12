import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Cấu hình giới hạn ---
MAX_IMAGES_PER_LABEL = 200  # Giới hạn ảnh chụp cho mỗi nhãn

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Tham số cắt ảnh
offset = 20
imgSize = 300

# Thư mục dữ liệu
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

assigned_labels = ["A", "B", "C", "D", "E", "F", "G"]
#assigned_labels = ["H", "I", "J", "K", "L", "M", "N"]   Châm
#assigned_labels = ["O", "P", "Q", "R", "S", "T", "U"]  Huyền
#assigned_labels = ["V", "W", "X", "Y", "Z", "del", "space"] Quỳnh

# Tạo thư mục cho từng nhãn nếu chưa có
for label in assigned_labels:
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

# Biến trạng thái
current_label_idx = 0
counter = len(os.listdir(os.path.join(DATA_DIR, assigned_labels[current_label_idx])))  # Đếm số ảnh đã có

print(f"Bắt đầu thu thập dữ liệu cho: {assigned_labels[current_label_idx]}")
print("Nhấn 's' để chụp ảnh, 'n' để chuyển nhãn tiếp theo, 'q' để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    imgWhite = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            x_min, y_min = max(0, x_min - offset), max(0, y_min - offset)
            x_max, y_max = min(w, x_max + offset), min(h, y_max + offset)
            imgCrop = frame[y_min:y_max, x_min:x_max]

            if imgCrop.size == 0:
                continue

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            h_crop, w_crop, _ = imgCrop.shape
            aspectRatio = h_crop / w_crop

            if aspectRatio > 1:
                k = imgSize / h_crop
                wCal = int(k * w_crop)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w_crop
                hCal = int(k * h_crop)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # Hiển thị thông tin lên khung hình
    cv2.putText(frame, f"Label: {assigned_labels[current_label_idx]} | Images: {counter}/{MAX_IMAGES_PER_LABEL}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == ord("s"):
        if counter < MAX_IMAGES_PER_LABEL and imgWhite is not None:
            counter += 1
            filename = f"{counter}_{time.time()}.jpg"
            cv2.imwrite(os.path.join(DATA_DIR, assigned_labels[current_label_idx], filename), imgWhite)
            print(f"Đã lưu ảnh cho {assigned_labels[current_label_idx]} | Tổng: {counter}")
        elif counter >= MAX_IMAGES_PER_LABEL:
            print(f"Đã đủ {MAX_IMAGES_PER_LABEL} ảnh cho nhãn {assigned_labels[current_label_idx]}")

    elif key == ord("n"):
        current_label_idx = (current_label_idx + 1) % len(assigned_labels)
        label_dir = os.path.join(DATA_DIR, assigned_labels[current_label_idx])
        counter = len(os.listdir(label_dir))
        print(f"\nChuyển sang nhãn: {assigned_labels[current_label_idx]} (đã có {counter}/{MAX_IMAGES_PER_LABEL} ảnh)")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
