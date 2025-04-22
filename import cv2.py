import cv2
import dlib
import numpy as np

def overlay_image(bg, overlay, x, y, overlay_size=None):
    h, w = overlay.shape[:2]
    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size)
    for i in range(overlay.shape[0]):
        for j in range(overlay.shape[1]):
            if overlay[i,j][3] != 0:
                if 0 <= y+i < bg.shape[0] and 0 <= x+j < bg.shape[1]:
                    bg[y+i, x+j] = overlay[i,j][:3]
    return bg

# Load dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load filters
glasses_img = cv2.imread("filters/glasses.png", cv2.IMREAD_UNCHANGED)
mustache_img = cv2.imread("filters/moustache.png", cv2.IMREAD_UNCHANGED)
import os

# Load all filter paths
filter_files = [
    "filters/sunglasses.png",
    "filters/moustache.png",
    "filters/top_hat.png"
]
filters = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in filter_files]
current_filter = 0

# In your loop, use the current filter:
frame = overlay_image(frame, filters[current_filter], x, y, (size_w, size_h))

# Switch filter with key press
key = cv2.waitKey(1) & 0xFF
if key == ord('n'):
    current_filter = (current_filter + 1) % len(filters)
elif key == ord('q'):
    break


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

        # Glasses between eyes (points 36 and 45)
        left_eye = points[36]
        right_eye = points[45]
        eye_width = int(np.linalg.norm(np.array(left_eye) - np.array(right_eye)) * 2)
        x = int((left_eye[0] + right_eye[0]) / 2) - eye_width // 2
        y = int((left_eye[1] + right_eye[1]) / 2) - eye_width // 3
        frame = overlay_image(frame, glasses_img, x, y, (eye_width, int(eye_width*0.4)))

        # Mustache below nose (between points 33 and 51)
        nose = points[33]
        mustache_width = 60
        frame = overlay_image(frame, mustache_img, nose[0] - mustache_width//2, nose[1] + 10, (mustache_width, 30))

    cv2.imshow("AR Filter - dlib version", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
