import cv2
import dlib
import numpy as np

# Load dlib's detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def overlay_image(bg, overlay, x, y, overlay_size=None):
    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size)
    h, w = overlay.shape[:2]
    for i in range(h):
        for j in range(w):
            if overlay[i, j][3] != 0 and y+i < bg.shape[0] and x+j < bg.shape[1]:
                bg[y+i, x+j] = overlay[i, j][:3]
    return bg

# Load all filters from /filters/
import os
filter_dir = "filters"
filter_files = [f for f in os.listdir(filter_dir) if f.endswith(".png")]
filters = {f.replace(".png", ""): cv2.imread(os.path.join(filter_dir, f), cv2.IMREAD_UNCHANGED) for f in filter_files}
filter_keys = list(filters.keys())
current_filter = 0

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

        filter_name = filter_keys[current_filter]
        overlay = filters[filter_name]

        if filter_name == "glasses" or filter_name == "eye_mask":
            left_eye = points[36]
            right_eye = points[45]
            center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            width = int(np.linalg.norm(np.array(left_eye) - np.array(right_eye)) * 2)
            x = center[0] - width // 2
            y = center[1] - width // 3
        elif filter_name == "moustache" or filter_name == "rabbit":
            nose = points[33]
            x = nose[0] - 30
            y = nose[1] + 10
            width = 60
        elif filter_name == "hat" or filter_name == "clown_hat":
            forehead = points[27]
            x = forehead[0] - 60
            y = forehead[1] - 130
            width = 120
        else:
            x, y, width = 100, 100, 80  # fallback position

        frame = overlay_image(frame, overlay, x, y, (width, int(width * overlay.shape[0] / overlay.shape[1])))

    cv2.putText(frame, f"Filter: {filter_keys[current_filter]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("AR Filter (dlib)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        current_filter = (current_filter + 1) % len(filter_keys)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
