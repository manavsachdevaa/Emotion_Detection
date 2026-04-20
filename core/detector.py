import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from deepface import DeepFace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "../emotion_model.h5")

cnn_model = load_model(model_path, compile=False)

emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

prediction_history = []
frame_count = 0
last_prediction = None  # 🔥 cache result


def process_frame(frame):
    global prediction_history, frame_count, last_prediction

    frame_count += 1

    # 🔥 Resize smaller (faster)
    frame_small = cv2.resize(frame, (480, 360))

    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.2, 6)

    for (x, y, w, h) in faces:

        # 🔥 Skip processing most frames
        if frame_count % 3 != 0 and last_prediction is not None:
            emotion, confidence = last_prediction
        else:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (64, 64))
            roi = roi / 255.0
            roi = np.reshape(roi, (1, 64, 64, 1))

            cnn_pred = cnn_model.predict(roi, verbose=0)[0]

            # 🔥 DeepFace only rarely
            if frame_count % 10 == 0:
                try:
                    face_img = frame_small[y:y+h, x:x+w]
                    result = DeepFace.analyze(
                        face_img,
                        actions=['emotion'],
                        detector_backend='opencv',
                        enforce_detection=False
                    )

                    if isinstance(result, list):
                        result = result[0]

                    df_emotions = result["emotion"]

                    deepface_pred = np.array([
                        df_emotions[label] for label in emotion_labels
                    ]) / 100

                except:
                    deepface_pred = cnn_pred
            else:
                deepface_pred = cnn_pred

            final_pred = (0.7 * cnn_pred) + (0.3 * deepface_pred)

            prediction_history.append(final_pred)
            if len(prediction_history) > 5:   # 🔥 smaller buffer = faster
                prediction_history.pop(0)

            avg_pred = np.mean(prediction_history, axis=0)

            max_index = np.argmax(avg_pred)
            confidence = avg_pred[max_index] * 100
            emotion = emotion_labels[max_index].capitalize()

            last_prediction = (emotion, confidence)

        # 🔥 Draw
        if last_prediction:
            emotion, confidence = last_prediction

            if confidence > 35:
                label = f"{emotion} ({confidence:.1f}%)"

                cv2.rectangle(frame_small, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame_small, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return frame_small