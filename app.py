# 1. Imports
import streamlit as st
from PIL import Image
import cv2

from helmet_detection import detect_helmet, detect_video, detect_webcam


# 2. Page settings
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)


# 3. Title
st.title(" 🚵 Helmet Detection System For Road Safety")

st.markdown("""
## Welcome to the Helmet Detection System

This application detects whether a motorcyclist is wearing a helmet using
Artificial Intelligence and Computer Vision.

The system uses the **YOLOv8 Object Detection Model** to identify riders
wearing helmets and riders without helmets from:

- 📷 Images
- 🎥 Videos
- 📹 Live Webcam
""")

st.divider()

# ---------------- ABOUT PROJECT ----------------

st.header("📌 About the Project")

st.write("""
The objective of this project is to improve road safety by automatically
detecting helmet violations.

The model is trained on a custom Helmet Detection Dataset and provides
real-time detection with high accuracy.
""")

st.subheader("This project can be used by:")

st.write("""
👮 Traffic Police

🏙 Smart City Projects

🎓 Educational Purposes

🔬 Computer Vision Research
""")

st.divider()

# ---------------- TECHNOLOGIES ----------------

st.header("🛠 Technologies Used")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("🐍 Python")
    st.success("👁 YOLOv8")
    st.success("📷 OpenCV")

with col2:
    st.success("🌐 Streamlit")
    st.success("🔥 PyTorch")
    st.success("📊 NumPy")

with col3:
    st.success("💻 VS Code")
    st.success("☁ Google Colab")


    st.divider()

st.header("🤖 Machine Learning Model")

st.subheader("Model Used")

st.info("YOLOv8 Nano")

st.divider()

st.header("📂 Dataset Information")

st.write("""
Dataset contains:

🪖 Riders With Helmet

❌ Riders Without Helmet

Dataset preprocessing includes:

✔ Image Resizing

✔ Annotation Verification

✔ Data Splitting

✔ Label Validation
""")

st.divider()

st.header("🎯 Classes Detected")

c1, c2 = st.columns(2)

with c1:
    st.success("🪖 With Helmet")

with c2:
    st.error("❌ Without Helmet")


st.divider()


# 4. Image Detection section
st.header("📷 Image Detection")


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Original Image"
    )


    if st.button("Detect Helmet"):

        detected_image, helmet_count, no_helmet_count, confidence = detect_helmet(image)


        st.image(
            detected_image,
            caption="Detected Image"
        )


        st.success(
            f"🪖 Helmet Count: {helmet_count}"
        )


        st.error(
            f"❌ No Helmet Count: {no_helmet_count}"
        )


        st.info(
            f"🎯 Confidence: {confidence*100:.2f}%"
        )



# 6. Video Detection

st.header("🎥 Video Detection")


video_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)

if video_file:

    video_path = "input_video.mp4"

    with open(video_path, "wb") as f:
        f.write(video_file.read())


    if st.button("Detect Video"):

        output_video = detect_video(video_path)

        st.video(output_video)


# ---------------- LIVE WEBCAM ----------------

st.header("📹 Live Webcam Detection")

run_webcam = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run_webcam:

    success, frame = camera.read()

    if not success:
        st.error("Unable to access webcam.")
        break

    frame = detect_webcam(frame)

    FRAME_WINDOW.image(
        frame,
        channels="BGR"
    )

camera.release()

st.divider()

# ---------------- FEATURES ----------------

st.header("✨ Features")

col1, col2 = st.columns(2)

with col1:
    st.success("📷 Upload Image")
    st.success("🎥 Upload Video")
    st.success("📹 Live Webcam Detection")

with col2:
    st.success("⚡ Real-Time Prediction")
    st.success("🎯 High-Speed Detection")
    st.success("🖥 Easy User Interface")


st.divider()

# ---------------- CONCLUSION ----------------

st.header("📝 Conclusion")

st.write("""
The Helmet Detection System is an AI-powered application that automatically
detects whether motorcycle riders are wearing helmets.

Using the YOLOv8 object detection model, the system performs
real-time helmet detection with high accuracy.

It can assist traffic authorities, smart city projects,
and researchers in improving road safety and reducing
helmet rule violations.
""")


st.divider()

# ---------------- FOOTER ----------------

st.markdown(
"""
<center>

### 🪖 Helmet Detection System

Developed using **YOLOv8 | Streamlit | OpenCV | PyTorch**

© 2026 Helmet Detection Project

</center>
""",
unsafe_allow_html=True
)