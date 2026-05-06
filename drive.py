import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="wide"
)

# ---------------- ANIMATED BACKGROUND + STYLE ----------------
st.markdown("""
<style>

/* Animated Background */
.stApp {
    background: linear-gradient(-45deg, #dbeafe, #f0f9ff, #e0f2fe, #bfdbfe);
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
}

/* Background Animation */
@keyframes gradientBG {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}

/* Floating Title Animation */
@keyframes floatTitle {
    0% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-8px);
    }
    100% {
        transform: translateY(0px);
    }
}

/* Main Title */
h1 {
    text-align: center;
    color: #0f172a;
    font-weight: bold;
}

/* Subtitle */
h3 {
    text-align: center;
    color: #334155;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f8fafc;
}

/* Buttons */
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.05);
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background: white;
    border-radius: 15px;
    padding: 15px;
    border: 1px solid #dbeafe;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.06);
}

</style>
""", unsafe_allow_html=True)

# ---------------- FIXED TITLE BOX ----------------
st.markdown("""
<h1 style='
text-align: center;
color: #0f172a;
font-size: 42px;
font-weight: bold;
animation: floatTitle 3s ease-in-out infinite;
'>
🚗 Driver Drowsiness Detection System
</h1>

<h3 style='
text-align: center;
color: #334155;
'>
😴 Stay Alert • Drive Safe
</h3>
""", unsafe_allow_html=True)

# ---------------- LOAD CASCADES ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ---------------- SIDEBAR ----------------
option = st.sidebar.radio(
    "Choose Input Type",
    ["Upload Image", "Webcam"]
)

status = "No Detection Yet"
active = 0
sleep = 0
faces = []

# ---------------- IMAGE MODE ----------------
if option == "Upload Image":

    file = st.file_uploader(
        "📤 Upload Driver Image",
        type=["jpg", "jpeg", "png"]
    )

    if file is not None:
        img = Image.open(file)
        frame = np.array(img)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        color = (0, 255, 0)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) == 0:
                status = "😴 Drowsy Detected"
                color = (0, 0, 255)
                sleep += 1
            else:
                status = "😊 Driver Active"
                color = (0, 255, 0)
                active += 1

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

        st.image(frame, channels="BGR", width=500)
        st.success(status)

# ---------------- WEBCAM MODE ----------------
elif option == "Webcam":

    run = st.checkbox("▶ Start Live Camera")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        photo_saved = False

        while run:
            ret, frame = cap.read()

            if not ret:
                st.error("❌ Camera not working")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            status = "😊 Active"
            color = (0, 255, 0)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)

                if len(eyes) == 0:
                    sleep += 1

                    if sleep > 10:
                        status = "😴 Drowsiness Detected"
                        color = (0, 0, 255)

                        if not photo_saved:
                            cv2.imwrite("alert.jpg", frame)
                            photo_saved = True

                            st.error("🚨 Driver is Sleeping!")

                            # Small Captured Image
                            st.image(
                                "alert.jpg",
                                caption="📸 Captured Alert Image",
                                width=300
                            )

                else:
                    sleep = 0
                    active += 1
                    status = "😊 Driver Active"
                    color = (0, 255, 0)
                    photo_saved = False

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

            FRAME_WINDOW.image(
                frame,
                channels="BGR",
                width=700
            )

        cap.release()

# ---------------- DASHBOARD ----------------
st.subheader("📊 Live Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("👤 Faces Detected", len(faces))

with col2:
    st.metric("😴 Driver Status", status)

with col3:
    st.metric("📷 Detection Mode", option)

# ---------------- PIE CHART ----------------
st.subheader("🥧 Driver Monitoring Pie Chart")

labels = ["😊 Active", "😴 Drowsy"]
sizes = [active, sleep]

if active == 0 and sleep == 0:
    sizes = [1, 1]

fig, ax = plt.subplots(figsize=(5, 5))

ax.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

ax.axis("equal")
st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>Travel Safe!!❤️ </center>",
    unsafe_allow_html=True
)