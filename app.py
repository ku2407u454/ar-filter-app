import streamlit as st
import cv2
import dlib
import numpy as np
import os
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load dlib detector and predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# Load filters from /filters directory
def load_filters():
    filters = {}
    for file in os.listdir("filters"):
        if file.endswith(".png"):
            key = file.split(".")[0]
            filters[key] = cv2.imread(os.path.join("filters", file), cv2.IMREAD_UNCHANGED)
    return filters

FILTERS = load_filters()

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    if overlay is None:
        return background

    if overlay_size is not None:
        overlay = cv2.resize(overlay.copy(), overlay_size)

    b_h, b_w = background.shape[:2]
    if x >= b_w or y >= b_h:
        return background

    h, w = overlay.shape[:2]
    if x + w > b_w:
        w = b_w - x
        overlay = overlay[:, :w]
    if y + h > b_h:
        h = b_h - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        return background

    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_img
    return background

class ARFilterTransformer(VideoTransformerBase):
    def __init__(self):
        self.selected_filter = "glasses"
        self.effect = "None"
        self.capture = False
        self.frame_to_save = None
        self.face_count = 0

    def set_filter(self, selected_filter):
        self.selected_filter = selected_filter

    def set_effect(self, effect):
        self.effect = effect

    def trigger_capture(self):
        self.capture = True

    def apply_effect(self, frame):
        if self.effect == "Grayscale":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.effect == "Cartoon":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(frame, 9, 300, 300)
            return cv2.bitwise_and(color, color, mask=edges)
        return frame

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.apply_effect(img) if self.effect != "None" else img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        self.face_count = len(faces)

        for face in faces:
            landmarks = predictor(gray, face)
            points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

            if self.selected_filter in FILTERS:
                overlay = FILTERS[self.selected_filter]
                if self.selected_filter in ["glasses", "eye_mask"]:
                    x1, y1 = points[36]
                    x2, y2 = points[45]
                    width = int(np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) * 2)
                    x = int((x1 + x2) / 2) - width // 2
                    y = int((y1 + y2) / 2) - width // 3
                elif self.selected_filter in ["moustache", "rabbit"]:
                    x, y = points[33][0] - 30, points[33][1] + 10
                    width = 60
                else:
                    x, y = points[27][0] - 60, points[27][1] - 130
                    width = 120

                img = overlay_transparent(img, overlay, x, y, (width, int(width * overlay.shape[0] / overlay.shape[1])))

        if self.capture:
            self.frame_to_save = img.copy()
            cv2.imwrite("captured_filter_frame.png", self.frame_to_save)
            self.capture = False

        return img

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🎝️ Real-Time AR Filter App (dlib)")
st.sidebar.title("🛠️ Controls")

selected_filter = st.sidebar.selectbox("Choose Filter", list(FILTERS.keys()))
effect_name = st.sidebar.selectbox("Apply Effect", ["None", "Grayscale", "Cartoon"])

st.sidebar.markdown("### Filter Previews")
cols = st.sidebar.columns(3)
for i, (name, f_img) in enumerate(FILTERS.items()):
    if f_img is not None:
        with cols[i % 3]:
            st.image(f"filters/{name}.png", caption=name, width=80)

ctx = webrtc_streamer(key="ar-dlib", video_transformer_factory=ARFilterTransformer, async_transform=True)

if ctx.video_transformer:
    ctx.video_transformer.set_filter(selected_filter)
    ctx.video_transformer.set_effect(effect_name)

    if st.button("📸 Capture Frame"):
        ctx.video_transformer.trigger_capture()
        st.success("Saved as captured_filter_frame.png")

    st.markdown(f"🧠 Faces detected: {ctx.video_transformer.face_count}")
