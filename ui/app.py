import streamlit as st
import cv2
import sys
import os
import time

# ================= FIX IMPORT PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, PARENT_DIR)

from core.detector import process_frame

# ================= UI CONFIG =================
st.set_page_config(page_title="Emotion Detector", layout="centered")

st.title("😊 Real-Time Emotion Detection")
st.write("Hybrid Model (Fast + Accurate)")

# ================= CONTROLS =================
run = st.checkbox("Start Camera")

frame_window = st.image([])

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

# ================= MAIN LOOP =================
if run:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Camera not working")
            break

        # 🔥 Process frame using your hybrid model
        processed_frame = process_frame(frame)

        # 🔥 Show in UI
        frame_window.image(processed_frame, channels="BGR")

        # 🔥 CONTROL FPS (VERY IMPORTANT)
        time.sleep(0.03)   # ~30 FPS → smooth + low lag

else:
    st.info("👆 Click 'Start Camera' to begin")

# ================= CLEANUP =================
cap.release()