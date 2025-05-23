# 🎭 Real-Time AR Filter App

Apply fun and dynamic AR filters to your face in real-time using dlib and Streamlit! This app uses face landmark detection and overlays transparent PNG filters like glasses, moustaches, hats, and more — all inside your browser using your webcam.

[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-orange?logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)](https://opencv.org/)

---

## 🚀 Live Demo

👉 Launch the app on [Streamlit Cloud](https://streamlit.io/cloud)  
Once deployed, your app will appear at:
```
https://<your-username>-ar-filter-app.streamlit.app
```

---

## 📸 Features

- 🧠 Real-time facial landmark tracking using dlib
- 🎭 AR overlays: glasses, hats, eye masks, clown nose, and more
- 🎨 Visual effects: Grayscale, Cartoon
- 👤 Face detection count display
- 📷 Capture and save filtered webcam images
- 👁️‍🗨️ Filter previews in sidebar

---

## 🛠 How to Run Locally

1. Clone this repository:
```bash
git clone https://github.com/ku2407u454/ar-filter-app.git
cd ar-filter-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dlib model (if not already in folder):
🔗 https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2  
Extract it to:
```
shape_predictor_68_face_landmarks.dat
```

4. Run the app:
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                        # Streamlit app
├── filters/                     # Transparent PNG AR filters
├── requirements.txt             # Required packages
├── shape_predictor_68_face_landmarks.dat
└── README.md
```

---

## 📸 Example Filters

| Glasses | Clown Hat | Rabbit |
|--------|------------|--------|
| ![Glasses](filters/glasses.png) | ![Clown](filters/clown_hat.png) | ![Rabbit](filters/rabbit.png) |

---

## 💡 Credits

- Built with 🧠 dlib, 💻 OpenCV, and 🎨 Streamlit
- AR filter idea by [your name here]

---

## 📬 Contact

Got questions or ideas? Open an issue or ping me on GitHub!