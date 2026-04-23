# Emotion Detection using Deep Learning
# This file handles real-time emotion detection using DeepFace and OpenCV
import cv2
import numpy as np
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

last_emotion = "neutral"
last_confidence = 0
frame_count = 0

emotion_history = []

# Processes each video frame and detects dominant emotion
def process_frame(frame):
    global last_emotion, last_confidence, frame_count, emotion_history

    frame_count += 1
    print("Processing new frame...")

    frame_small = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    # Detect faces in grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        print("No face detected")
        return frame_small

    (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])

    if frame_count % 4 == 0:
        face_img = frame_small[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False
            )

            emotion_data = result[0]['emotion']
            emotion_data['angry'] *= 1.2
            emotion_data['surprise'] *= 1.15
            emotion_history.append(emotion_data)

            if len(emotion_history) > 5:
                emotion_history.pop(0)

        except:
            pass

    if len(emotion_history) == 0:
        return frame_small
    # Calculate average emotions over last few frames for stability
    avg_emotions = {}

    for key in emotion_history[0].keys():
        avg_emotions[key] = np.mean([e[key] for e in emotion_history])

    last_emotion = max(avg_emotions, key=avg_emotions.get)
    last_confidence = avg_emotions[last_emotion]
    if last_confidence > 25:
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
    print(f"Detected emotion: {last_emotion} with confidence {last_confidence}")
    return frame_small
