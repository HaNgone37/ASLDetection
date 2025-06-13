import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import gdown

# ==== CẤU HÌNH ====
MODEL_ID = "1-qtMLem63El7msIK84PzMmMTvgyr9T1_"  # Google Drive file ID
MODEL_PATH = "hand_sign_cnn_model.h5"
IMG_SIZE = 224
LABELS = [
    'A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'space'
]

# ==== TẢI MODEL ====
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

# ==== TIỀN XỬ LÝ ẢNH ====
def preprocess_image(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Tăng tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Làm nét
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, sharp_kernel)

    # Resize và chuẩn hóa
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # shape: (1, 224, 224, 1)
    return img

# ==== GIAO DIỆN ====
st.set_page_config(page_title="Nhận diện ký hiệu tay", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #ff4b4b;'>Nhận diện Ký hiệu Tay bằng CNN</h1>
    <p style='text-align: center;'>Tải lên ảnh ký hiệu tay để dự đoán chữ cái. Ứng dụng sử dụng TensorFlow + Streamlit Cloud.</p>
    """, unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader("Tải ảnh ký hiệu tay (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Ảnh bạn đã chọn", width=300)
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Tiền xử lý
    input_tensor = preprocess_image(img_bgr)

    # Load model và dự đoán
    model = load_model()
    prediction = model.predict(input_tensor)[0]
    pred_index = np.argmax(prediction)
    confidence = prediction[pred_index]

    # Hiển thị kết quả
    st.markdown(f"""
        <h2 style='text-align:center;'>
            Dự đoán: <span style='color:#4CAF50'>{LABELS[pred_index]}</span> 
            (Độ tin cậy: {confidence:.2f})
        </h2>
        """, unsafe_allow_html=True)

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(LABELS, prediction, color='skyblue')
    bars[pred_index].set_color('green')
    ax.set_title("Phân bố xác suất dự đoán", fontsize=14)
    ax.set_ylabel("Xác suất")
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45)
    st.pyplot(fig)
