import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import threading
import queue
import time

model = tf.keras.models.load_model("sign_languge_model.keras")

label_encoder = sorted(os.listdir("./sign_language_data"))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

font_path = "C:/Windows/Fonts/arial.ttf"
if not os.path.exists(font_path):
    font = ImageFont.load_default()
else:
    font = ImageFont.truetype(font_path, 32)

def normalize_landmarks(landmarks):
    wrist = np.array(landmarks[0])
    palm_center = np.mean([landmarks[5], landmarks[9], landmarks[13], landmarks[17]], axis=0)
    base_distance = np.linalg.norm(wrist - palm_center)
    return (landmarks - wrist) / base_distance if base_distance > 0 else np.zeros_like(landmarks)

sequence = []

def detect_sign_from_frame(img):
    global sequence

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    hands_data = []

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            hand_points = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
            norm_points = normalize_landmarks(hand_points)
            hands_data.append(norm_points.flatten())

    if len(hands_data) == 1:
        hands_data.append(np.zeros(63))
    elif len(hands_data) == 0:
        hands_data = [np.zeros(63), np.zeros(63)]

    if len(hands_data) == 2:
        data = np.concatenate(hands_data)
        if data.shape == (126,):
            sequence.append(data)
            if len(sequence) > 30:
                sequence.pop(0)

            if len(sequence) == 30:
                input_data = np.expand_dims(np.array(sequence), axis=0)
                input_data = (input_data - 0.5) * 2

                prediction = model.predict(input_data, verbose=0)
                predicted_index = np.argmax(prediction)
                confidence = np.max(prediction)

                if confidence > 0.9:
                    return f"{label_encoder[predicted_index]} ({confidence:.2%})"
                else:
                    return "Không rõ"
    return "Đang xử lý..."

def draw_label_on_frame(frame, label):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((20, 20), label, font=font, fill=(255, 0, 0))
    return np.array(img_pil)

frame_queue = queue.Queue(maxsize=50)
result_label = "Đang xử lý..."
lock = threading.Lock()

def frame_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được khung hình.")
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        time.sleep(0.01) 
    cap.release()

def frame_predict():
    global result_label, sequence
    skip_frame_count = 0
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            skip_frame_count = (skip_frame_count + 1) % 3
            if skip_frame_count != 0:
                with lock:
                    pass
            else:
                label = detect_sign_from_frame(frame)
                with lock:
                    result_label = label

def run_detection():
    threading.Thread(target=frame_capture, daemon=True).start()
    threading.Thread(target=frame_predict, daemon=True).start()

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            with lock:
                label = result_label
            frame = draw_label_on_frame(frame, label)
            cv2.imshow("Phát hiện ngôn ngữ ký hiệu", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()