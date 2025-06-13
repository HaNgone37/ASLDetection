import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# ==== CẤU HÌNH ====
MODEL_PATH = "hand_sign_cnn_model.h5"
IMG_SIZE = 224
LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space'
]

model = tf.keras.models.load_model(MODEL_PATH)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
offset = 20
imgSize = 300

class SignApp:
    def __init__(self, root):
        self.root = root
        root.title("Nhận diện ký hiệu tay (Webcam) - Ghép chữ")
        root.geometry("800x600")

        self.video_label = Label(root)
        self.video_label.pack()

        self.text_display = Label(root, text="", font=("Helvetica", 24), fg="blue")
        self.text_display.pack(pady=10)

        self.reset_button = Button(root, text="🗑️ Xóa toàn bộ", command=self.reset_text)
        self.reset_button.pack()

        self.accumulated_text = ""
        self.last_char = ""
        self.confirm_count = 0
        self.confirm_threshold = 15  # số frame liên tiếp giống nhau để chấp nhận ký tự

        self.cap = cv2.VideoCapture(0)
        self.update_frame()

        root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def preprocess_crop(self, img_crop):
        img_white = np.ones((imgSize, imgSize, 3), dtype=np.uint8) * 255
        h, w, _ = img_crop.shape
        aspect_ratio = h / w

        if aspect_ratio > 1:
            k = imgSize / h
            w_cal = int(k * w)
            img_resize = cv2.resize(img_crop, (w_cal, imgSize))
            w_gap = (imgSize - w_cal) // 2
            img_white[:, w_gap:w_gap + w_cal] = img_resize
        else:
            k = imgSize / w
            h_cal = int(k * h)
            img_resize = cv2.resize(img_crop, (imgSize, h_cal))
            h_gap = (imgSize - h_cal) // 2
            img_white[h_gap:h_gap + h_cal, :] = img_resize

        gray = cv2.cvtColor(img_white, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(contrast, -1, sharp_kernel)
        resized = cv2.resize(sharp, (IMG_SIZE, IMG_SIZE))
        norm = resized.astype(np.float32) / 255.0
        return np.expand_dims(norm, axis=(0, -1))

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)  # Xoay ngược như gương

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                x_min, y_min = max(0, x_min - offset), max(0, y_min - offset)
                x_max, y_max = min(w, x_max + offset), min(h, y_max + offset)
                img_crop = frame[y_min:y_max, x_min:x_max]
                if img_crop.size == 0:
                    return

                input_tensor = self.preprocess_crop(img_crop)
                prediction = model.predict(input_tensor)[0]
                pred_idx = np.argmax(prediction)
                confidence = prediction[pred_idx]
                pred_label = LABELS[pred_idx]

                # Chỉ lấy ký tự khi confidence cao
                if confidence > 0.8:
                    if pred_label == self.last_char:
                        self.confirm_count += 1
                    else:
                        self.confirm_count = 0
                    self.last_char = pred_label

                    if self.confirm_count == self.confirm_threshold:
                        if pred_label == "space":
                            self.accumulated_text += " "
                        elif pred_label == "del":
                            self.accumulated_text = self.accumulated_text[:-1]
                        else:
                            self.accumulated_text += pred_label
                        self.text_display.config(text=self.accumulated_text)
                        self.confirm_count = 0  # reset sau khi thêm ký tự

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{pred_label} ({confidence:.2f})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def reset_text(self):
        self.accumulated_text = ""
        self.text_display.config(text="")

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignApp(root)
    root.mainloop()
