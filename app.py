import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from pathlib import Path
import os
from collections import deque, Counter
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "asl_model.h5"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")

# Load your trained model
model = load_model(str(MODEL_PATH))
MODEL_INPUT_SHAPE = model.input_shape
MODEL_NUM_CLASSES = int(model.output_shape[-1])
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.70"))
SMOOTHING_WINDOW = int(os.getenv("SMOOTHING_WINDOW", "8"))
prediction_buffer = deque(maxlen=max(SMOOTHING_WINDOW, 1))
SHOW_TOPK = int(os.getenv("SHOW_TOPK", "3"))
MISS_RESET_FRAMES = int(os.getenv("MISS_RESET_FRAMES", "6"))
miss_count = 0
LIVE_PREDICT = os.getenv("LIVE_PREDICT", "0") == "1"
PREPROCESS_MODES = ["BGR", "RGB", "GRAY3", "GRAY_BIN3"]
NORM_MODES = ["ZERO_ONE", "NEG_ONE_ONE"]
preprocess_mode = os.getenv("PREPROCESS_MODE", "RGB").upper()
if preprocess_mode not in PREPROCESS_MODES:
    preprocess_mode = "RGB"
norm_mode = os.getenv("NORM_MODE", "ZERO_ONE").upper()
if norm_mode not in NORM_MODES:
    norm_mode = "ZERO_ONE"
SAVE_CAPTURES = os.getenv("SAVE_CAPTURES", "1") == "1"
CAPTURE_DIR = BASE_DIR / "captures"
if SAVE_CAPTURES:
    CAPTURE_DIR.mkdir(exist_ok=True)
last_hand_img = None
snap_text = ""
snap_topk = []
snap_frames_left = 0

# Common 36-class mapping used by many ASL/alphanumeric datasets.
# Primary default: A-Z + 0-9
DEFAULT_36_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
# Alternate if your training used 0-9 + A-Z
ALT_36_LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
if MODEL_NUM_CLASSES == 36:
    labels = ALT_36_LABELS if os.getenv("LABEL_ORDER") == "DIGITS_FIRST" else DEFAULT_36_LABELS
elif MODEL_NUM_CLASSES == 26:
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
else:
    labels = [f"Class_{i}" for i in range(MODEL_NUM_CLASSES)]

def _make_square_crop(img, x_min, y_min, x_max, y_max, pad=0.2):
    h, w, _ = img.shape
    bw = max(x_max - x_min, 1)
    bh = max(y_max - y_min, 1)
    side = int(max(bw, bh) * (1.0 + pad))
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    sx_min = max(cx - side // 2, 0)
    sy_min = max(cy - side // 2, 0)
    sx_max = min(sx_min + side, w)
    sy_max = min(sy_min + side, h)
    return img[sy_min:sy_max, sx_min:sx_max], sx_min, sy_min, sx_max, sy_max


def predict_letter(hand_img):
    # Expected format: (None, H, W, C)
    _, in_h, in_w, in_c = MODEL_INPUT_SHAPE
    in_h = int(in_h or 64)
    in_w = int(in_w or 64)
    in_c = int(in_c or 3)

    img = cv2.resize(hand_img, (in_w, in_h))

    if in_c == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(1, in_h, in_w, 1)
    else:
        if preprocess_mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif preprocess_mode == "GRAY3":
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        elif preprocess_mode == "GRAY_BIN3":
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (5, 5), 0)
            _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        img = img.reshape(1, in_h, in_w, 3)

    img = img.astype("float32")
    if norm_mode == "NEG_ONE_ONE":
        img = (img / 127.5) - 1.0
    else:
        img = img / 255.0
    prediction = model.predict(img, verbose=0)
    index = int(np.argmax(prediction))
    confidence = float(prediction[0][index])
    label = labels[index] if index < len(labels) else f"Class_{index}"
    topk = np.argsort(prediction[0])[::-1][:max(SHOW_TOPK, 1)]
    topk_items = []
    for i in topk:
        idx = int(i)
        topk_label = labels[idx] if idx < len(labels) else f"Class_{idx}"
        topk_items.append((topk_label, float(prediction[0][idx])))
    return label, confidence, topk_items


use_mediapipe_solutions = hasattr(mp, "solutions")
hands = None
mp_hands = None
mp_draw = None

if use_mediapipe_solutions:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not access camera. Check webcam permissions or camera index.")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_for_model = frame.copy()
    h, w, _ = frame.shape

    if use_mediapipe_solutions:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_seen = False
            for hand_landmarks in result.multi_hand_landmarks:
                hand_seen = True
                x_list = []
                y_list = []

                for lm in hand_landmarks.landmark:
                    x_list.append(int(lm.x * w))
                    y_list.append(int(lm.y * h))

                x_min = max(min(x_list) - 20, 0)
                x_max = min(max(x_list) + 20, w)
                y_min = max(min(y_list) - 20, 0)
                y_max = min(max(y_list) + 20, h)

                if x_max <= x_min or y_max <= y_min:
                    continue

                hand_img, sx_min, sy_min, sx_max, sy_max = _make_square_crop(
                    frame_for_model, x_min, y_min, x_max, y_max, pad=0.25
                )
                cv2.rectangle(frame, (sx_min, sy_min), (sx_max, sy_max), (0, 255, 0), 2)

                if hand_img.size != 0:
                    last_hand_img = hand_img.copy()
                    if LIVE_PREDICT:
                        letter, confidence, topk_items = predict_letter(hand_img)
                        if confidence >= MIN_CONFIDENCE:
                            prediction_buffer.append(letter)
                            miss_count = 0
                        else:
                            miss_count += 1
                        if miss_count >= MISS_RESET_FRAMES:
                            prediction_buffer.clear()
                        if prediction_buffer:
                            stable_letter = Counter(prediction_buffer).most_common(1)[0][0]
                        else:
                            stable_letter = "..."
                        cv2.putText(
                            frame,
                            f"{stable_letter} ({letter}) {confidence:.2f}",
                            (x_min, max(y_min - 10, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 255, 0),
                            3
                        )
                        if SHOW_TOPK > 0:
                            debug_text = " | ".join([f"{k}:{v:.2f}" for k, v in topk_items[:SHOW_TOPK]])
                            cv2.putText(
                                frame,
                                debug_text,
                                (20, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2
                            )
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
            if not hand_seen:
                miss_count += 1
                if miss_count >= MISS_RESET_FRAMES:
                    prediction_buffer.clear()
        else:
            miss_count += 1
            if miss_count >= MISS_RESET_FRAMES:
                prediction_buffer.clear()
    else:
        # Fallback mode for mediapipe builds without `mp.solutions`.
        box_w = int(w * 0.45)
        box_h = int(h * 0.6)
        x_min = (w - box_w) // 2
        y_min = (h - box_h) // 2
        x_max = x_min + box_w
        y_max = y_min + box_h

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        hand_img = frame[y_min:y_max, x_min:x_max]
        if hand_img.size != 0:
            last_hand_img = hand_img.copy()
        if hand_img.size != 0 and LIVE_PREDICT:
            letter, confidence, _ = predict_letter(hand_img)
            cv2.putText(
                frame,
                f"{letter} {confidence:.2f} (ROI)",
                (x_min, max(y_min - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                2
            )
        cv2.putText(
            frame,
            "Mediapipe landmarks unavailable. Place hand in box.",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

    cv2.putText(
        frame,
        f"live={'on' if LIVE_PREDICT else 'off'} mode={preprocess_mode} norm={norm_mode} | keys: c=mode n=norm p=snap r=reset esc=exit",
        (10, h - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )
    if snap_frames_left > 0:
        cv2.putText(
            frame,
            snap_text,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 200, 255),
            3
        )
        if snap_topk:
            snap_debug = " | ".join([f"{k}:{v:.2f}" for k, v in snap_topk[:SHOW_TOPK]])
            cv2.putText(
                frame,
                snap_debug,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2
            )
        snap_frames_left -= 1
    cv2.imshow("Sign Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        preprocess_mode = PREPROCESS_MODES[(PREPROCESS_MODES.index(preprocess_mode) + 1) % len(PREPROCESS_MODES)]
        prediction_buffer.clear()
        miss_count = 0
    elif key == ord("n"):
        norm_mode = NORM_MODES[(NORM_MODES.index(norm_mode) + 1) % len(NORM_MODES)]
        prediction_buffer.clear()
        miss_count = 0
    elif key == ord("r"):
        prediction_buffer.clear()
        miss_count = 0
    elif key == ord("p"):
        if last_hand_img is not None and last_hand_img.size != 0:
            snap_label, snap_conf, snap_topk = predict_letter(last_hand_img)
            snap_text = f"SNAP: {snap_label} {snap_conf:.2f}"
            snap_frames_left = 120
            if SAVE_CAPTURES:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = CAPTURE_DIR / f"{stamp}_{snap_label}.png"
                cv2.imwrite(str(out_path), last_hand_img)
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
