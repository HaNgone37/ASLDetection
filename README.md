# Hệ Thống Nhận Diện Ký Hiệu ASL

**Lưu ý:** Đây là bài tập môn Trí Tuệ Nhân Tạo. Mọi dữ liệu đều được tự thu thập, xử lý với quy mô nhỏ và chỉ phục vụ mục đích học tập. 

## Mô tả

Đây là dự án nhỏ nhận diện ký hiệu **American Sign Language (ASL)** bằng Python. Hệ thống bao gồm:
- Thu thập dữ liệu hình ảnh tự thu.
- Tiền xử lý dữ liệu, gán nhãn.
- Huấn luyện mô hình CNN.
- Demo nhận diện qua giao diện Streamlit và Tkinter.

## Kiến trúc thư mục

- `data_collection.py` — Thu thập hình ảnh ký hiệu bằng webcam.
- `preprocess_dataset.py` — Xử lý ảnh, resize, chuẩn hoá dữ liệu.
- `train_model.ipynb` — Huấn luyện mô hình CNN.
- `streamlit_app.py` — Giao diện web demo real-time bằng Streamlit.
- `tkinter_gui.py` — Giao diện desktop demo bằng Tkinter.
- `requirements.txt` — Thư viện Python cần cài.
- `README.md` — File mô tả dự án.

## Link model huấn luyện

[Tải model tại đây](https://drive.google.com/file/d/1-qtMLem63El7msIK84PzMmMTvgyr9T1_/view?usp=drive_link)

## Cách chạy

1. Tạo môi trường Python:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
2. Cài đặt thư viện
   pip install -r requirements.txt
3. Chạy Tkinter hoặc Streamlit
