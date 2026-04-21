import cv2
import numpy as np
from deepface import DeepFace

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= GLOBAL =================
last_emotion = "Neutral"
last_confidence = 0
frame_count = 0

def process_frame(frame):
    global last_emotion, last_confidence, frame_count

    frame_count += 1

    # 🔥 Resize for speed
    frame_small = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return frame_small

    # Largest face
    (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])

    # 🔥 Run DeepFace only every 5 frames (IMPORTANT for speed)
    if frame_count % 5 == 0:
        face_img = frame_small[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False
            )

            emotion_data = result[0]['emotion']

            # Get dominant emotion
            last_emotion = max(emotion_data, key=emotion_data.get)
            last_confidence = emotion_data[last_emotion]

        except:
            pass  # ignore errors safely

    # 🔥 Draw result
    if last_confidence > 30:
        label = f"{last_emotion.capitalize()} ({last_confidence:.1f}%)"

        cv2.rectangle(frame_small, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(
            frame_small,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    return frame_small