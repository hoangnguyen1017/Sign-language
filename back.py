import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import threading
import queue
import time
import unicodedata

# === Load model ===
model = tf.keras.models.load_model("sign_languge_model.keras")
label_encoder = sorted(os.listdir("./sign_language_data"))

# === MediaPipe config ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === Font vẽ ===
font_path = "C:/Windows/Fonts/arial.ttf"
font = ImageFont.truetype(font_path, 32) if os.path.exists(font_path) else ImageFont.load_default()

# === Bộ đệm & kết quả ===
sequence = []
frame_queue = queue.Queue(maxsize=50)
result_label = "Đang xử lý..."
result_text = ""
lock = threading.Lock()

# ==== Tiền xử lý landmarks ====
def normalize_landmarks(landmarks):
    wrist = np.array(landmarks[0])
    palm_center = np.mean([landmarks[5], landmarks[9], landmarks[13], landmarks[17]], axis=0)
    base_distance = np.linalg.norm(wrist - palm_center)
    return (landmarks - wrist) / base_distance if base_distance > 0 else np.zeros_like(landmarks)

# ==== Xử lý text ====
reverse_replacements = {
    "slash": "/", "backslash": "\\", "question": "?",
    "tilde": "~", "dot": "."
}

accent_map = {
    ("A", "^"): "Â", ("E", "^"): "Ê", ("O", "^"): "Ô", ("U", "'"): "Ư", ("O", "'"): "Ơ", ("A", "w"): "Ă",
    ("A", "."): "Ạ", ("A", "/"): "Á", ("A", "\\"): "À", ("A", "~"): "Ã", ("A", "?"): "Ả",
    ("E", "."): "Ẹ", ("E", "/"): "É", ("E", "\\"): "È", ("E", "~"): "Ẽ", ("E", "?"): "Ẻ",
    ("O", "."): "Ọ", ("O", "/"): "Ó", ("O", "\\"): "Ò", ("O", "~"): "Õ", ("O", "?"): "Ỏ",
    ("U", "."): "Ụ", ("U", "/"): "Ú", ("U", "\\"): "Ù", ("U", "~"): "Ũ", ("U", "?"): "Ủ",
    ("I", "."): "Ị", ("I", "/"): "Í", ("I", "\\"): "Ì", ("I", "~"): "Ĩ", ("I", "?"): "Ỉ",
    ("Y", "."): "Ỵ", ("Y", "/"): "Ý", ("Y", "\\"): "Ỳ", ("Y", "~"): "Ỹ", ("Y", "?"): "Ỷ",
    ("Â", "."): "Ậ", ("Â", "/"): "Ấ", ("Â", "\\"): "Ầ", ("Â", "~"): "Ẫ", ("Â", "?"): "Ẩ",
    ("Ê", "."): "Ệ", ("Ê", "/"): "Ế", ("Ê", "\\"): "Ề", ("Ê", "~"): "Ễ", ("Ê", "?"): "Ể",
    ("Ô", "."): "Ộ", ("Ô", "/"): "Ố", ("Ô", "\\"): "Ồ", ("Ô", "~"): "Ỗ", ("Ô", "?"): "Ổ",
    ("Ơ", "."): "Ợ", ("Ơ", "/"): "Ớ", ("Ơ", "\\"): "Ờ", ("Ơ", "~"): "Ỡ", ("Ơ", "?"): "Ở",
    ("Ư", "."): "Ự", ("Ư", "/"): "Ứ", ("Ư", "\\"): "Ừ", ("Ư", "~"): "Ữ", ("Ư", "?"): "Ử",
    ("Ă", "."): "Ặ", ("Ă", "/"): "Ắ", ("Ă", "\\"): "Ằ", ("Ă", "~"): "Ẵ", ("Ă", "?"): "Ẳ"
}

def decode_label(label):
    return reverse_replacements.get(label, label)

def combine_vietnamese_characters():
    global result_text
    if len(result_text) >= 2:
        last_two = (result_text[-2], result_text[-1])
        if last_two in accent_map:
            result_text = result_text[:-2] + accent_map[last_two]

# ==== Public API cho web.py dùng ====
def get_result_text():
    return result_text

def clear_result_text():
    global result_text
    result_text = ""

def save_result_text(filename="saved_text.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(result_text)

def append_result_text(label):
    global result_text
    decoded = decode_label(label)
    result_text += decoded
    combine_vietnamese_characters()

# ==== Nhận diện ký hiệu từ ảnh ====
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
                    label = label_encoder[predicted_index]
                    append_result_text(label)
                    return f"{label} ({confidence:.2%})"
                else:
                    return "Không rõ"
    return "Đang xử lý..."

# ==== Vẽ label lên frame ====
def draw_label_on_frame(frame, label):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((20, 20), label, font=font, fill=(255, 0, 0))
    return np.array(img_pil)

# ==== Thread (nếu cần chạy offline demo) ====
def frame_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được camera.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        time.sleep(0.01)
    cap.release()

def frame_predict():
    global result_label
    skip = 0
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            skip = (skip + 1) % 3
            if skip != 0:
                continue
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