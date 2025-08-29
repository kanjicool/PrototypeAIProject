import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

st.title("Image Processing with Streamlit")

# 1. เลือกแหล่งภาพ
source = st.radio("เลือกแหล่งภาพ", ["Webcam", "URL"])

img = None
if source == "Webcam":
    img_file = st.camera_input("ถ่ายภาพด้วยเว็บแคม")
    if img_file is not None:
        img = Image.open(img_file)
elif source == "URL":
    url = st.text_input("ใส่ URL ของรูปภาพ")
    if url:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"ไม่สามารถโหลดรูปภาพ: {e}")

if img is not None:
    st.image(img, caption="Original Image", use_column_width=True)

    # 2. Custom parameter สำหรับ image processing
    st.sidebar.header("Image Processing Parameters")
    blur_ksize = st.sidebar.slider("Blur Kernel Size", 1, 25, 5, step=2)
    canny_low = st.sidebar.slider("Canny Threshold Low", 0, 255, 50)
    canny_high = st.sidebar.slider("Canny Threshold High", 0, 255, 150)

    # 3. Image Processing
    img_np = np.array(img.convert("RGB"))
    img_blur = cv2.GaussianBlur(img_np, (blur_ksize, blur_ksize), 0)
    img_canny = cv2.Canny(img_blur, canny_low, canny_high)

    st.image(img_canny, caption="Processed Image (Canny Edge)", use_column_width=True, channels="GRAY")

    # 4. แสดงกราฟจากคุณสมบัติของรูปภาพ (Histogram)
    st.subheader("Histogram of Processed Image")
    fig, ax = plt.subplots()
    ax.hist(img_canny.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    ax.set_title("Pixel Value Histogram")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
else:
    st.info("กรุณาเลือกรูปภาพจาก Webcam หรือ URL")