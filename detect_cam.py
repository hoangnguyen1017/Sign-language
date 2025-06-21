import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import unicodedata

# === Load model ===
model_path = "sign_languge_model.keras"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("✅ Đã tải mô hình")
else:
    print("❌ Không tìm thấy mô hình:", model_path)
    exit()

# === Label encoder ===
label_encoder = sorted(os.listdir("sign_language_data"))
reverse_replacements = {
    "slash": "/", "backslash": "\\", "question": "?", "tilde": "~", "dot": "."
}

# === Font ===
font_path = "C:/Windows/Fonts/arial.ttf"
try:
    font = ImageFont.truetype("arial.ttf", 32)
except OSError:
    print("⚠️ Không tìm thấy font 'arial.ttf', dùng font mặc định.")
    font = ImageFont.load_default()

# === Mediapipe ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# === Bộ đệm chuỗi dự đoán ===
result_text = ""
delete_running = False
delete_active = False
last_prediction= None
sequence = []
frame_threshold = 30
append_cooldown = False


# ==== Hàm xử lý ====
def normalize_landmarks(landmarks):
    wrist = np.array(landmarks[0])
    palm_center = np.mean([landmarks[5], landmarks[9], landmarks[13], landmarks[17]], axis=0)
    base_distance = np.linalg.norm(wrist - palm_center)
    return (landmarks - wrist) / base_distance if base_distance > 0 else np.zeros_like(landmarks)

def decode_label(label):
    return reverse_replacements.get(label, label)

def remove_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# ==== Dấu tiếng Việt ====
accent_map = {
    ("A", "^"): "Â", ("E", "^"): "Ê", ("O", "^"): "Ô", ("U", "'"): "Ư", ("O", "'"): "Ơ",
    ("A", "w"): "Ă", ("A", "."): "Ạ", ("A", "/"): "Á", ("A", "\\"): "À", ("A", "~"): "Ã", ("A", "?"): "Ả",
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

def combine_vietnamese_characters():
    global result_text

    if not result_text or (result_text[-1] in {".", "/", "\\", "~", "?"} and len(result_text) == 1):
        result_text = result_text[:-1]
        return

    if len(result_text) >= 2 and result_text[-1] in {".", "/", "\\", "~", "?"}:
        if result_text[-2] not in {key[0] for key in accent_map}:
            result_text = result_text[:-1]
            return

    if len(result_text) >= 2:
        last_two = (result_text[-2], result_text[-1])
        if last_two in accent_map:
            result_text = result_text[:-2] + accent_map[last_two]
            return

    if len(result_text) >= 3:
        last_three = (result_text[-3], result_text[-2], result_text[-1])
        if last_three in accent_map:
            result_text = result_text[:-3] + accent_map[last_three]
            return
        
def delete_last_character():
    global result_text, delete_running, delete_active
    if result_text:
        print(f"Deleting character. Current text: {result_text}")
        result_text = result_text[:-1]
        delete_running = False
        delete_active = True
    else:
        print("No more characters to delete.")
        delete_running = False
        delete_active = True

def update_result_box(new_text, append=True):
    global result_text, delete_running, delete_active

    if not append:
        result_text = ""
        delete_running = False
        return

    prev_text = result_text

    if new_text == "space":
        result_text += " "
        delete_running = False

    elif new_text == "del":
        if not delete_running:
            delete_running = True
            delete_last_character()

    else:
        result_text += new_text
        delete_running = False

    if result_text != prev_text:
        combine_vietnamese_characters()

def draw_label_on_frame(frame, label):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((20, 50), label, font=font, fill=(0, 255, 0))
    return np.array(img_pil)

def reset_append_cooldown():
    global append_cooldown
    append_cooldown = False
def get_result_text():
    global result_text
    return result_text
# ==== Dự đoán và cập nhật văn bản ====
def detect_sign_from_frame(frame):
    global sequence, last_prediction, result_text, delete_running, delete_active, append_cooldown

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    frame_hands_data = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_data = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            norm_hand_data = normalize_landmarks(hand_data)
            frame_hands_data.append(norm_hand_data.flatten())
        if len(frame_hands_data) == 1:
            frame_hands_data.append(np.zeros(63))
    else:
        frame_hands_data = [np.zeros(63), np.zeros(63)]

    frame_hands_data = np.concatenate(frame_hands_data)

    if frame_hands_data.shape == (126,):
        sequence.append(frame_hands_data)

        if len(sequence) == frame_threshold:
            input_data = np.array(sequence)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = (input_data - 0.5) * 2

            prediction = model.predict(input_data, verbose=0)
            predicted_label = label_encoder[np.argmax(prediction)]
            confidence = np.max(prediction)
            sequence.clear()

            if confidence >= 0.9:
                decoded_label = decode_label(predicted_label)

                if decoded_label != last_prediction or not append_cooldown:
                    update_result_box(decoded_label, append=True)
                    last_prediction = decoded_label
                    append_cooldown = True

                    # Reset cooldown sau 1 giây (nếu bạn dùng tkinter)
                    try:
                        threading.Timer(1000, lambda: reset_append_cooldown())
                    except:
                        import threading
                        threading.Timer(1.0, reset_append_cooldown).start()

                    return decoded_label

    return ""

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở webcam.")
        exit()

    print("📸 Bắt đầu nhận diện ký hiệu... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không thể đọc frame từ webcam.")
            break

        label = detect_sign_from_frame(frame)
        if label:
            print("✅ Nhận diện:", label, "| Văn bản:", result_text)

        frame_display = draw_label_on_frame(frame, result_text)
        cv2.imshow("🖐️ Nhận diện ngôn ngữ ký hiệu (Ấn Q để thoát)", frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("📴 Đã dừng webcam.")