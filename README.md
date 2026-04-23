# 🎭 Emotion Detection using Deep Learning

This project is a **facial emotion detection system** built using deep learning. It analyzes human facial expressions and predicts emotions such as **happy, sad, angry, surprised, etc.**

---

## 🚀 Features

* Detect emotions from images
* Pre-trained deep learning model (`emotion_model.h5`)
* Simple UI interface (inside `ui` folder)
* Modular code structure (`core` + `ui`)
* Easy to run locally

---

## 🧠 Tech Stack

* Python 🐍
* TensorFlow / Keras
* OpenCV
* NumPy

---

## 📂 Project Structure

```
Emotion_Detection/
│
├── core/                 # Core logic (model + prediction)
├── ui/                   # User interface
├── emotion_model.h5      # Pre-trained model
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/manavsachdevaa/Emotion_Detection.git
cd Emotion_Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

(If needed)

```bash
pip install tensorflow==2.10.0 keras==2.10.0 --force-reinstall
```

---

## ▶️ How to Run

Run the main file (check inside `ui` or `core` for the entry file, usually something like):

```bash
python app.py
```

or

```bash
python main.py
```

---

## 📊 Model Details

* Model file: `emotion_model.h5`
* Trained using CNN (Convolutional Neural Network)
* Input: Facial image
* Output: Emotion label

---

## 🖼️ Example Emotions Detected

* Happy 😊
* Sad 😢
* Angry 😠
* Surprise 😲
* Neutral 😐

---

## 🛠️ Future Improvements

* Real-time webcam detection
* Better UI (web app)
* Higher accuracy model
* Deployment (Streamlit / Flask)

---

## 🤝 Contributing

Feel free to fork this repo and submit pull requests.

---

## 📜 License

This project is open-source and free to use.

---

## 👨‍💻 Author

Manav Sachdeva,
Kratika Gupta,
Kishan Kumar Kasaudhan,
Harsh Vardhan Srivastava

---

⭐ If you like this project, give it a star!
