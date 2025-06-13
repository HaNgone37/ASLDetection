import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# === THIẾT LẬP ===
DATA_DIR = "./data"  # Thư mục chứa các thư mục con: A/, B/, C/, ...
IMG_SIZE = 224

# === TIỀN XỬ LÝ 1 ẢNH ===
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Cải thiện độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Làm nét
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    img = cv2.filter2D(img, -1, sharpen_kernel)

    # Resize & Chuẩn hóa
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  #shape: (224,224,1)
    return img

# === TẢI VÀ TIỀN XỬ LÝ TOÀN BỘ TẬP DỮ LIỆU ===
def load_dataset(data_dir):
    images, labels = [], []
    label_names = sorted(os.listdir(data_dir))
    label_dict = {name: idx for idx, name in enumerate(label_names)}

    for label_name in label_names:
        label_path = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(label_path, fname)
                try:
                    img = preprocess_image(fpath)
                    images.append(img)
                    labels.append(label_dict[label_name])
                except:
                    print(f"Không xử lý được ảnh: {fpath}")

    X = np.array(images)
    y = np.array(labels)
    return X, y, label_names

# === CHẠY TIỀN XỬ LÝ + TÁCH DỮ LIỆU ===
if __name__ == "__main__":
    X, y, label_names = load_dataset(DATA_DIR)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Hiển thị ảnh mẫu
    plt.imshow(X_train[0].squeeze(), cmap="gray")
    plt.title(f"Label: {label_names[y_train[0]]}")
    plt.axis('off')
    plt.show()

    print("Dữ liệu đã tiền xử lý xong:")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")