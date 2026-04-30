"""
Flask Web Application for Sign Language to Text Conversion.
Replicates 100% of the functionality from final_pred.py in a web-based UI.
"""

from dotenv import load_dotenv
load_dotenv()  # Load .env file before reading os.environ

import numpy as np
import math
import cv2
import os
import sys
import traceback
import threading
import time
import json
import base64
from functools import wraps
import pyttsx3
import mediapipe as mp
import requests
from flask import Flask, render_template, Response, jsonify, request, redirect, session, url_for
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
try:
    import enchant
    ddd = enchant.Dict("en-US")
except ImportError:
    enchant = None
    ddd = None
    print("WARNING: enchant not available, spell-check disabled")
from googletrans import Translator
from gtts import gTTS

translator = Translator()

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "sign-language-web-secret")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGIN_USERNAME = os.environ.get("APP_USERNAME", "admin")
LOGIN_PASSWORD = os.environ.get("APP_PASSWORD", "admin123")
ACTION_PROJECT_PATH = os.path.join(BASE_DIR, "ActionDetectionforSignLanguage-main")
ACTION_MODEL_PATH = os.path.join(ACTION_PROJECT_PATH, "action.h5")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
ACTION_SEQUENCE_LENGTH = 30
ACTION_STABILITY_WINDOW = 10
ACTION_THRESHOLD = 0.5

# Detect if running on cloud (Render sets RENDER=true, or check for PORT env)
CLOUD_MODE = bool(os.environ.get('RENDER') or os.environ.get('IS_CLOUD'))
if CLOUD_MODE:
    print("☁️  Running in CLOUD MODE — browser webcam will be used")

# ---------------------------------------------------------------------------
# Globals (mirrors final_pred.py Application class attributes)
# ---------------------------------------------------------------------------
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings

# ddd already initialized above in the try/except block
hd = HandDetector(maxHands=1)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
offset = 29
cap = None
vs = None
current_camera_idx = 0
# Load model
model = load_model('cnn8grps_rad1_model.h5')
print("Loaded model from disk")
action_model = None
if os.path.exists(ACTION_MODEL_PATH):
    action_model = load_model(ACTION_MODEL_PATH)
    print("Loaded action model")
else:
    print("Skipping action model (not found)")
action_labels = np.array(['hello', 'thanks', 'iloveyou'])
action_colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# TTS engine — created per-speak to avoid cross-thread issues
speak_lock = threading.Lock()

# State variables (same as Application.__init__)
ct = {}
ct['blank'] = 0
for i in ascii_uppercase:
    ct[i] = 0

blank_flag = 0
space_flag = False
next_flag = True
prev_char = ""
count = -1
ten_prev_char = [" "] * 10
str_output = " "
ccc = 0
word = " "
current_symbol = "C"
word1 = " "
word2 = " "
word3 = " "
word4 = " "
pts = []

# Camera — auto-detect and track available cameras
camera_lock = threading.Lock()
current_camera_idx = 0

# 🔹 after imports and global vars


def reset_state():
    global str_output, action_sentence, vs

    str_output = ""
    action_sentence = []

    # 🔥 RESET CAMERA
    if vs is not None:
        vs.release()
        vs = None

    print("State reset + camera reset")


def get_camera():
    global vs, current_camera_idx

    if vs is None or not vs.isOpened():
        vs = cv2.VideoCapture(current_camera_idx)
        print(f"Camera started at index {current_camera_idx}")

    return vs

def discover_cameras(max_index=5):
    """Probe camera indices and return list of available cameras."""
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # Try to read a frame to confirm it's real
            ret, _ = cap.read()
            if ret:
                name = f"Camera {idx}"
                # Try to get backend name
                backend = cap.getBackendName() if hasattr(cap, 'getBackendName') else 'Unknown'
                available.append({'index': idx, 'name': f"Camera {idx} ({backend})"})
            cap.release()
    return available

available_cameras = []
if not CLOUD_MODE:
    available_cameras = discover_cameras()
    print(f"Available cameras: {available_cameras}")
else:
    print("☁️  Skipping camera discovery (cloud mode)")

# Open first available camera
vs = None
if not CLOUD_MODE:
    if available_cameras:
        current_camera_idx = available_cameras[0]['index']
        print(f"Using camera index {current_camera_idx}")
    else:
        vs = cv2.VideoCapture(1)
        current_camera_idx = 0
        print("No camera detected, defaulting to index 0")
else:
    print("☁️  No server camera needed (browser webcam mode)")

def switch_camera(new_idx):
    """Switch to a different camera index at runtime."""
    global vs, current_camera_idx

    with camera_lock:
        try:
            # 🔥 create new camera properly
            new_cap = cv2.VideoCapture(new_idx)

            if new_cap.isOpened():
                ret, _ = new_cap.read()

                if ret:
                    # 🔥 release old camera
                    if vs is not None:
                        vs.release()

                    # 🔥 assign new camera
                    vs = new_cap
                    current_camera_idx = new_idx

                    print(f"Switched to camera index {new_idx}")
                    return True

            # ❌ if failed
            new_cap.release()
            return False

        except Exception as e:
            print("Switch camera error:", e)
            return False
# Frame buffers for streaming
latest_camera_frame = None
latest_skeleton_frame = None
frame_lock = threading.Lock()
processing = True
app_status = "starting"
app_error = ""
active_mode = "alphabet"
camera_rotation = 0  # 0, 90, 180, 270 degrees

action_sequence = []
action_sentence = []
action_predictions = []
action_current_symbol = "-"
action_probabilities = [0.0 for _ in action_labels]
analysis_result = ""
analysis_status = "idle"


# ---------------------------------------------------------------------------
# Helper: distance (same as Application.distance)
# ---------------------------------------------------------------------------
def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


def normalize_hands_result(result):
    """
    cvzone HandDetector.findHands returns either:
    - (hands, img) on newer versions
    - hands on some older variants
    final_pred.py was written against a structure where hands[0][0] is the hand dict.
    This helper accepts both shapes and returns a plain list of hand dicts.
    """
    if isinstance(result, tuple):
        hands = result[0]
    else:
        hands = result

    if hands is None:
        return []

    if not isinstance(hands, list):
        return []

    normalized = []
    for item in hands:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, (list, tuple)) and item and isinstance(item[0], dict):
            normalized.append(item[0])

    return normalized


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_action_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )


def extract_action_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


def action_prob_viz(res, labels, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 180), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, f"{labels[num]} {prob:.2f}", (10, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame


def is_logged_in():
    return session.get("authenticated", False)


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if is_logged_in():
            return view_func(*args, **kwargs)

        if request.path.startswith("/api/") or request.path in {"/state", "/action", "/speak", "/clear", "/video_feed", "/skeleton_feed"}:
            return jsonify({"ok": False, "error": "Authentication required"}), 401

        return redirect(url_for("login"))

    return wrapped_view


def set_active_mode(mode):
    global active_mode
    active_mode = mode


def reset_action_state():
    global action_sequence, action_sentence, action_predictions, action_current_symbol, action_probabilities
    action_sequence = []
    action_sentence = []
    action_predictions = []
    action_current_symbol = "-"
    action_probabilities = [0.0 for _ in action_labels]


def speak_text_value(text_value, lang="EN"):
    if not text_value or not text_value.strip():
        return

    try:
        lang_map = {
            "EN": "en",
            "TA": "ta",
            "TE": "te"
        }

        target_lang = lang_map.get(lang, "en")

        if target_lang != "en":
            text_value = translator.translate(text_value, dest=target_lang).text

        tts = gTTS(text=text_value, lang=target_lang)
        tts.save("static/output.mp3")

    except Exception as e:
        print("TTS Error:", e)


# Cooldown tracker for analyze requests (prevents 429 rate limit spam)
_last_analyze_time = 0
ANALYZE_COOLDOWN_SECONDS = 5  # minimum seconds between analyze calls (Groq free: 30 RPM)


def analyze_frame_with_groq(frame):
    """Analyze a camera frame using Groq Vision API (OpenAI-compatible)."""
    if not GROQ_API_KEY:
        return "Groq analysis is not configured. Set GROQ_API_KEY in your .env file."

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return "Failed to capture the current frame for analysis."

    image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    prompt = (
        "You are assisting a blind user. Analyze this webcam image and say only the most useful result. "
        "If there is a visible hand sign, identify the sign or gesture. "
        "If there is readable text, read it. "
        "If neither is clear, briefly describe what is visible in one short sentence. "
        "Keep the answer concise and speech-friendly."
    )

    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    # Retry with exponential backoff for 429 rate-limit errors
    max_retries = 3
    for attempt in range(max_retries):
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=45,
        )

        if response.status_code == 429:
            wait_time = 2 ** (attempt + 1)
            print(f"[Groq] Rate limited (429). Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
            continue

        response.raise_for_status()
        data = response.json()

        # Extract text from Groq/OpenAI-compatible response
        try:
            text = data["choices"][0]["message"]["content"].strip()
            if text:
                return text
        except (KeyError, IndexError):
            pass

        return "The image was analyzed, but no spoken result was returned."

    return "Rate limited by Groq API. Please wait a few seconds and try again."


# ---------------------------------------------------------------------------
# Predict function — EXACT replica of Application.predict from final_pred.py
# ---------------------------------------------------------------------------
def predict(test_image):
    global str_output, word, word1, word2, word3, word4
    global current_symbol, prev_char, count, ten_prev_char, pts

    white = test_image
    white_input = white.reshape(1, 400, 400, 3)
    prob = np.array(model.predict(white_input, verbose=0)[0], dtype='float32')
    ch1 = np.argmax(prob, axis=0)
    prob[ch1] = 0
    ch2 = np.argmax(prob, axis=0)
    prob[ch2] = 0
    ch3 = np.argmax(prob, axis=0)
    prob[ch3] = 0

    pl = [ch1, ch2]

    # condition for [Aemnst]
    l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
         [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
         [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 0

    # condition for [o][s]
    l = [[2, 2], [2, 1]]
    if pl in l:
        if (pts[5][0] < pts[4][0]):
            ch1 = 0

    # condition for [c0][aemnst]
    l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
            ch1 = 2

    # condition for [c0][aemnst]
    l = [[6, 0], [6, 6], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) < 52:
            ch1 = 2

    # condition for [gh][bdfikruvw]
    l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 3

    # con for [gh][l]
    l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 3

    # con for [gh][pqz]
    l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[2][1] + 15 < pts[16][1]:
            ch1 = 3

    # con for [l][x]
    l = [[6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) > 55:
            ch1 = 4

    # con for [l][d]
    l = [[1, 4], [1, 6], [1, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) > 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 4

    # con for [l][gh]
    l = [[3, 6], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[0][0]):
            ch1 = 4

    # con for [l][c0]
    l = [[2, 2], [2, 5], [2, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[1][0] < pts[12][0]):
            ch1 = 4

    # con for [l][c0]
    l = [[2, 2], [2, 5], [2, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[1][0] < pts[12][0]):
            ch1 = 4

    # con for [gh][z]
    l = [[3, 6], [3, 5], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
            ch1 = 5

    # con for [gh][pq]
    l = [[3, 2], [3, 1], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][1] + 17 > pts[20][1]:
            ch1 = 5

    # con for [l][pqz]
    l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 5

    # con for [pqz][aemnst]
    l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 5

    # con for [pqz][yj]
    l = [[5, 7], [5, 2], [5, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[3][0] < pts[0][0]:
            ch1 = 7

    # con for [l][yj]
    l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] < pts[8][1]:
            ch1 = 7

    # con for [x][yj]
    l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] > pts[20][1]:
            ch1 = 7

    # condition for [x][aemnst]
    l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] > pts[16][0]:
            ch1 = 6

    # condition for [yj][x]
    l = [[7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
            ch1 = 6

    # condition for [c0][x]
    l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) > 50:
            ch1 = 6

    # con for [l][x]
    l = [[4, 6], [4, 2], [4, 1], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) < 60:
            ch1 = 6

    # con for [x][d]
    l = [[1, 4], [1, 6], [1, 0], [1, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 > 0:
            ch1 = 6

    # con for [b][pqz]
    l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
         [6, 3], [6, 4], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [f][pqz]
    l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
         [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [d][pqz]
    fg = 19
    l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
             pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[4][1] > pts[14][1]):
            ch1 = 1

    l = [[4, 1], [4, 2], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) < 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 1

    l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
             pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[14][1] < pts[4][1]):
            ch1 = 1

    l = [[6, 6], [6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 < 0:
            ch1 = 1

    # con for [i][pqz]
    l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
             pts[18][1] > pts[20][1])):
            ch1 = 1

    # con for [yj][bfdi]
    l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[5][0] + 15) and (
            (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
             pts[18][1] > pts[20][1])):
            ch1 = 7

    # con for [uvr]
    l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and
             pts[18][1] < pts[20][1])) and pts[4][1] > pts[14][1]:
            ch1 = 1

    # con for [w]
    fg = 13
    l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if not (pts[0][0] + fg < pts[8][0] and pts[0][0] + fg < pts[12][0] and pts[0][0] + fg < pts[16][0] and
                pts[0][0] + fg < pts[20][0]) and not (
                pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and distance(pts[4], pts[11]) < 50:
            ch1 = 1

    # con for [w]
    l = [[5, 0], [5, 5], [0, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]:
            ch1 = 1

    # -------------------------condn for 8 groups  ends
    # -------------------------condn for subgroups  starts

    if ch1 == 0:
        ch1 = 'S'
        if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
            ch1 = 'A'
        if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
            ch1 = 'T'
        if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
            ch1 = 'E'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
            ch1 = 'M'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
            ch1 = 'N'

    if ch1 == 2:
        if distance(pts[12], pts[4]) > 42:
            ch1 = 'C'
        else:
            ch1 = 'O'

    if ch1 == 3:
        if (distance(pts[8], pts[12])) > 72:
            ch1 = 'G'
        else:
            ch1 = 'H'

    if ch1 == 7:
        if distance(pts[8], pts[4]) > 42:
            ch1 = 'Y'
        else:
            ch1 = 'J'

    if ch1 == 4:
        ch1 = 'L'

    if ch1 == 6:
        ch1 = 'X'

    if ch1 == 5:
        if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
            if pts[8][1] < pts[5][1]:
                ch1 = 'Z'
            else:
                ch1 = 'Q'
        else:
            ch1 = 'P'

    if ch1 == 1:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'B'
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'D'
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'F'
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'I'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'W'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] < pts[9][1]:
            ch1 = 'K'
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'U'
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] > pts[9][1]):
            ch1 = 'V'
        if (pts[8][0] > pts[12][0]) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'R'

    if ch1 == 1 or ch1 == 'E' or ch1 == 'S' or ch1 == 'X' or ch1 == 'Y' or ch1 == 'B':
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = " "

    if ch1 == 'E' or ch1 == 'Y' or ch1 == 'B':
        if (pts[4][0] < pts[5][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = "next"

    if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and (pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1] and pts[4][1] < pts[16][1] and pts[4][1] < pts[20][1]) and (pts[4][1] < pts[6][1] and pts[4][1] < pts[10][1] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]):
            ch1 = 'Backspace'

    if ch1 == "next" and prev_char != "next":
        if ten_prev_char[(count - 2) % 10] != "next":
            if ten_prev_char[(count - 2) % 10] == "Backspace":
                str_output = str_output[0:-1]
            else:
                if ten_prev_char[(count - 2) % 10] != "Backspace":
                    str_output = str_output + ten_prev_char[(count - 2) % 10]
        else:
            if ten_prev_char[(count - 0) % 10] != "Backspace":
                str_output = str_output + ten_prev_char[(count - 0) % 10]

    if ch1 == "  " and prev_char != "  ":
        str_output = str_output + "  "

    prev_char = ch1
    current_symbol = ch1
    count += 1
    ten_prev_char[count % 10] = ch1

    # Word suggestions
    if len(str_output.strip()) != 0:
        st = str_output.rfind(" ")
        ed = len(str_output)
        w = str_output[st + 1:ed]
        word = w
        if len(w.strip()) != 0 and ddd is not None:
            ddd.check(w)
            suggestions = ddd.suggest(w)
            lenn = len(suggestions)
            word1 = suggestions[0] if lenn >= 1 else " "
            word2 = suggestions[1] if lenn >= 2 else " "
            word3 = suggestions[2] if lenn >= 3 else " "
            word4 = suggestions[3] if lenn >= 4 else " "
        else:
            word1 = word2 = word3 = word4 = " "


# ---------------------------------------------------------------------------
# Video processing loop (runs in background thread)
# ---------------------------------------------------------------------------
# def video_loop():
#     global latest_camera_frame, latest_skeleton_frame, pts, ccc, processing, app_status, app_error

#     while processing:
#         try:
#             cap = get_camera()
#             ok, frame = cap.read()
#             if not ok:
#                 app_status = "camera_error"
#                 app_error = "Failed to read from webcam."
#                 time.sleep(0.03)
#                 continue

#             cv2image = cv2.flip(frame, 1)
#             app_status = "ready"
#             app_error = ""

#             if cv2image is not None and cv2image.size > 0:
#                 hands = normalize_hands_result(hd.findHands(cv2image, draw=False, flipType=True))
#                 cv2image_copy = np.array(cv2image)

#                 # Encode camera frame for streaming
#                 with frame_lock:
#                     latest_camera_frame = cv2image.copy()

#                 if hands:
#                     hand = hands[0]
#                     x, y, w, h = hand['bbox']
                    
#                     # Ensure crop indices are valid
#                     y1 = max(0, y - offset)
#                     y2 = min(cv2image_copy.shape[0], y + h + offset)
#                     x1 = max(0, x - offset)
#                     x2 = min(cv2image_copy.shape[1], x + w + offset)
                    
#                     image = cv2image_copy[y1:y2, x1:x2]

#                     white = np.ones((400, 400, 3), dtype=np.uint8) * 255

#                     if image is not None and image.size > 0:
#                         handz = normalize_hands_result(hd2.findHands(image, draw=False, flipType=True))
#                         ccc += 1
#                         if handz:
#                             hand2 = handz[0]
#                             pts = hand2['lmList']

#                             os_x = ((400 - w) // 2) - 15
#                             os_y = ((400 - h) // 2) - 15

#                             # Draw skeleton lines — exact same as final_pred.py
#                             for t in range(0, 4, 1):
#                                 cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y),
#                                          (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
#                             for t in range(5, 8, 1):
#                                 cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y),
#                                          (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
#                             for t in range(9, 12, 1):
#                                 cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y),
#                                          (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
#                             for t in range(13, 16, 1):
#                                 cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y),
#                                          (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
#                             for t in range(17, 20, 1):
#                                 cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y),
#                                          (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
#                             cv2.line(white, (pts[5][0] + os_x, pts[5][1] + os_y),
#                                      (pts[9][0] + os_x, pts[9][1] + os_y), (0, 255, 0), 3)
#                             cv2.line(white, (pts[9][0] + os_x, pts[9][1] + os_y),
#                                      (pts[13][0] + os_x, pts[13][1] + os_y), (0, 255, 0), 3)
#                             cv2.line(white, (pts[13][0] + os_x, pts[13][1] + os_y),
#                                      (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)
#                             cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y),
#                                      (pts[5][0] + os_x, pts[5][1] + os_y), (0, 255, 0), 3)
#                             cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y),
#                                      (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)

#                             for i in range(21):
#                                 cv2.circle(white, (pts[i][0] + os_x, pts[i][1] + os_y), 2, (0, 0, 255), 1)

#                             res = white
#                             predict(res)

#                             with frame_lock:
#                                 latest_skeleton_frame = res.copy()
#                         else:
#                             with frame_lock:
#                                 latest_skeleton_frame = white.copy()
#                 else:
#                     with frame_lock:
#                         latest_skeleton_frame = np.ones((400, 400, 3), dtype=np.uint8) * 255

#         except Exception:
#             app_status = "processing_error"
#             app_error = traceback.format_exc().strip().splitlines()[-1]
#             print("==", traceback.format_exc())

#         time.sleep(0.01)  # ~30fps cap


# ---------------------------------------------------------------------------
# MJPEG streaming generators
# ---------------------------------------------------------------------------
def video_loop():
    global latest_camera_frame, latest_skeleton_frame, pts, ccc, processing, app_status, app_error
    global action_sequence, action_sentence, action_predictions, action_current_symbol, action_probabilities

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while processing:
            try:
                cap = get_camera()
                ok, frame = cap.read()
                if not ok:
                    app_status = "camera_error"
                    app_error = "Failed to read from webcam."
                    time.sleep(0.03)
                    continue

                cv2image = cv2.flip(frame, 1)

                # Apply rotation if set (for phone cameras)
                if camera_rotation == 90:
                    cv2image = cv2.rotate(cv2image, cv2.ROTATE_90_CLOCKWISE)
                elif camera_rotation == 180:
                    cv2image = cv2.rotate(cv2image, cv2.ROTATE_180)
                elif camera_rotation == 270:
                    cv2image = cv2.rotate(cv2image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                app_status = "ready"
                app_error = ""

                if cv2image is None or cv2image.size == 0:
                    time.sleep(0.01)
                    continue

                with frame_lock:
                    latest_camera_frame = cv2image.copy()

                if active_mode == "action":
                    processed_image, results = mediapipe_detection(cv2image.copy(), holistic)
                    draw_action_landmarks(processed_image, results)
                    keypoints = extract_action_keypoints(results)
                    action_sequence.append(keypoints)
                    action_sequence = action_sequence[-ACTION_SEQUENCE_LENGTH:]

                    if action_model is None:
                        action_current_symbol = "Model missing"
                        action_probabilities = [0.0 for _ in action_labels]
                    elif len(action_sequence) == ACTION_SEQUENCE_LENGTH:
                        res = action_model.predict(np.expand_dims(action_sequence, axis=0), verbose=0)[0]
                        action_probabilities = res.tolist()
                        predicted_index = int(np.argmax(res))
                        action_current_symbol = action_labels[predicted_index]
                        action_predictions.append(predicted_index)
                        action_predictions = action_predictions[-ACTION_STABILITY_WINDOW:]

                        stable_predictions = action_predictions[-ACTION_STABILITY_WINDOW:]
                        if (
                            len(stable_predictions) == ACTION_STABILITY_WINDOW
                            and len(set(stable_predictions)) == 1
                            and res[predicted_index] > ACTION_THRESHOLD
                        ):
                            predicted_label = action_labels[predicted_index]
                            if not action_sentence or predicted_label != action_sentence[-1]:
                                action_sentence.append(predicted_label)

                        if len(action_sentence) > 5:
                            action_sentence = action_sentence[-5:]

                        processed_image = action_prob_viz(res, action_labels, processed_image, action_colors)

                    cv2.rectangle(processed_image, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(processed_image, ' '.join(action_sentence), (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    with frame_lock:
                        latest_skeleton_frame = processed_image.copy()
                else:
                    hands = normalize_hands_result(hd.findHands(cv2image, draw=False, flipType=True))
                    cv2image_copy = np.array(cv2image)

                    if hands:
                        hand = hands[0]
                        x, y, w, h = hand['bbox']
                        y1 = max(0, y - offset)
                        y2 = min(cv2image_copy.shape[0], y + h + offset)
                        x1 = max(0, x - offset)
                        x2 = min(cv2image_copy.shape[1], x + w + offset)
                        image = cv2image_copy[y1:y2, x1:x2]
                        white = np.ones((400, 400, 3), dtype=np.uint8) * 255

                        if image is not None and image.size > 0:
                            handz = normalize_hands_result(hd.findHands(image, draw=False, flipType=True))
                            ccc += 1
                            if handz:
                                hand2 = handz[0]
                                pts = hand2['lmList']
                                os_x = ((400 - w) // 2) - 15
                                os_y = ((400 - h) // 2) - 15

                                for t in range(0, 4, 1):
                                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                                for t in range(5, 8, 1):
                                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                                for t in range(9, 12, 1):
                                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                                for t in range(13, 16, 1):
                                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                                for t in range(17, 20, 1):
                                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                                cv2.line(white, (pts[5][0] + os_x, pts[5][1] + os_y), (pts[9][0] + os_x, pts[9][1] + os_y), (0, 255, 0), 3)
                                cv2.line(white, (pts[9][0] + os_x, pts[9][1] + os_y), (pts[13][0] + os_x, pts[13][1] + os_y), (0, 255, 0), 3)
                                cv2.line(white, (pts[13][0] + os_x, pts[13][1] + os_y), (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)
                                cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y), (pts[5][0] + os_x, pts[5][1] + os_y), (0, 255, 0), 3)
                                cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y), (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)

                                for i in range(21):
                                    cv2.circle(white, (pts[i][0] + os_x, pts[i][1] + os_y), 2, (0, 0, 255), 1)

                                predict(white)

                            with frame_lock:
                                latest_skeleton_frame = white.copy()
                        else:
                            with frame_lock:
                                latest_skeleton_frame = white.copy()
                    else:
                        with frame_lock:
                            latest_skeleton_frame = np.ones((400, 400, 3), dtype=np.uint8) * 255

            except Exception:
                app_status = "processing_error"
                app_error = traceback.format_exc().strip().splitlines()[-1]
                print("==", traceback.format_exc())

            time.sleep(0.01)


def gen_camera():
    while True:
        with frame_lock:
            frame = latest_camera_frame
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)


def gen_skeleton():
    while True:
        with frame_lock:
            frame = latest_skeleton_frame
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            # Send white placeholder
            white = np.ones((400, 400, 3), dtype=np.uint8) * 255
            ret, jpeg = cv2.imencode('.jpg', white)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_logged_in():
        return redirect(url_for('index'))

    error = ""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
            session['authenticated'] = True
            session['username'] = username
            return redirect(url_for('index'))

        error = "Invalid username or password."

    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    set_active_mode("alphabet")
    return render_template('index1.html')


@app.route('/home')
@login_required
def home_redirect():
    return redirect(url_for('index', _anchor='home'))

@app.route('/speak', methods=['POST'])
@login_required
def speak():
    data = request.get_json()

    # 🔥 GET TEXT FROM FRONTEND (IMPORTANT)
    text = data.get('text')
    lang = data.get('lang', 'EN')

    # 🔥 fallback only if no text sent
    if not text:
        text = str_output if active_mode == "alphabet" else ' '.join(action_sentence)

    lang_map = {
        "EN": "en",
        "TA": "ta",
        "TE": "te"
    }

    target_lang = lang_map.get(lang, "en")

    try:
        from googletrans import Translator
        from gtts import gTTS
        import time

        translator = Translator()

        print("Original:", text)
        print("Lang:", target_lang)

        # 🔥 Translate safely
        if target_lang != "en":
            try:
                sentences = text.split('.')   # split long text
                translated_parts = []

                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    try:
                        t = translator.translate(s, dest=target_lang).text
                        translated_parts.append(t)
                    except:
                        translated_parts.append(s)

                text = '. '.join(translated_parts)

            except Exception as e:
                print("Translation error:", e)

        print("Final text:", text)

        # 🔥 prevent empty speech
        if not text.strip():
            return jsonify({'error': 'Empty text'})

        # 🔥 unique filename
        filename = f"output_{int(time.time())}.mp3"
        filepath = f"static/{filename}"

        tts = gTTS(text=text, lang=target_lang)
        tts.save(filepath)

        return jsonify({'audio': f'/{filepath}'})

    except Exception as e:
        print("TTS ERROR:", e)
        return jsonify({'error': str(e)})

@app.route('/translate')
@login_required
def translate_redirect():
    reset_state()
    reset_action_state()
    return render_template('index.html')

@app.route('/gestures')
@login_required
def gestures():
    return render_template('gestures.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/skeleton_feed')
def skeleton_feed():
    return Response(gen_skeleton(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/why-ishaara')
@login_required
def why_ishaara_redirect():
    return redirect(url_for('index', _anchor='why'))


@app.route('/blog')
@login_required
def blog_redirect():
    return redirect(url_for('index', _anchor='about'))


@app.route('/about')
@login_required
def about_redirect():
    return redirect(url_for('index', _anchor='about'))


@app.route('/dictionary')
@login_required
def dictionary_redirect():
    return redirect(url_for('index', _anchor='dictionary'))


@app.route('/models')
@login_required
def models_redirect():
    return redirect(url_for('action_detection'))


@app.route('/action-detection')
@login_required
def action_detection():
    set_active_mode("action")
    reset_state()
    return render_template('action.html')




def build_state():
    sentence_value = str_output if active_mode == "alphabet" else (' '.join(action_sentence) if action_sentence else " ")
    current_value = (str(current_symbol) if current_symbol else '-') if active_mode == "alphabet" else (action_current_symbol or '-')
    return {
        'mode': active_mode,
        'status': app_status,
        'error': app_error,
        'character': current_value,
        'current_symbol': current_value.strip() if isinstance(current_value, str) else current_value,
        'sentence': sentence_value,
        'word1': word1 if active_mode == "alphabet" else " ",
        'word2': word2 if active_mode == "alphabet" else " ",
        'word3': word3 if active_mode == "alphabet" else " ",
        'word4': word4 if active_mode == "alphabet" else " ",
        'word': word if active_mode == "alphabet" else " ",
        'suggestions': [word1.strip(), word2.strip(), word3.strip(), word4.strip()] if active_mode == "alphabet" else ["", "", "", ""],
        'action_labels': action_labels.tolist(),
        'action_probabilities': action_probabilities,
        'analysis_result': analysis_result,
        'analysis_status': analysis_status,
        'speak_available': True,
    }


@app.route('/state')
@app.route('/api/state')
@login_required
def get_state():
    """Return current application state as JSON — polled by the frontend."""
    return jsonify(build_state())


@app.route('/action', methods=['POST'])
@login_required
def action():
    """Handle suggestion button clicks — mirrors action1-4 from final_pred.py."""
    global str_output
    data = request.get_json()
    suggestion = data.get('suggestion', '').strip()
    if suggestion:
        idx_space = str_output.rfind(" ")
        idx_word = str_output.find(word, idx_space)
        str_output = str_output[:idx_word] + suggestion.upper()
    return jsonify({'sentence': str_output})


@app.route('/api/suggestion/<int:index>', methods=['POST'])
@login_required
def suggestion(index):
    global str_output
    suggestions = {
        1: word1,
        2: word2,
        3: word3,
        4: word4,
    }
    suggestion_value = suggestions.get(index, '').strip()
    if suggestion_value:
        idx_space = str_output.rfind(" ")
        idx_word = str_output.find(word, idx_space)
        str_output = str_output[:idx_word] + suggestion_value.upper()
    return jsonify({'sentence': str_output})


@app.route('/api/analyze-frame', methods=['POST'])
@login_required
def analyze_frame():
    global analysis_result, analysis_status, _last_analyze_time

    # Enforce cooldown to prevent 429 rate limits
    now = time.time()
    elapsed = now - _last_analyze_time
    if elapsed < ANALYZE_COOLDOWN_SECONDS:
        remaining = int(ANALYZE_COOLDOWN_SECONDS - elapsed) + 1
        return jsonify({'ok': False, 'error': f'Please wait {remaining}s before analyzing again.'}), 429

    with frame_lock:
        frame = None if latest_camera_frame is None else latest_camera_frame.copy()

    if frame is None:
        analysis_status = "error"
        analysis_result = "No live camera frame is available yet."
        return jsonify({'ok': False, 'error': analysis_result}), 400

    _last_analyze_time = time.time()
    analysis_status = "processing"
    try:
        analysis_text = analyze_frame_with_groq(frame)
        analysis_result = analysis_text
        analysis_status = "done"
        threading.Thread(target=speak_text_value, args=(analysis_text, "EN"), daemon=True).start()
        return jsonify({'ok': True, 'analysis_result': analysis_result})
    except Exception as exc:
        analysis_status = "error"
        analysis_result = f"Image analysis failed: {exc}"
        return jsonify({'ok': False, 'error': analysis_result}), 500


@app.route('/api/process-frame', methods=['POST'])
@login_required
def process_browser_frame():
    """Receive a base64 frame from the browser webcam, run hand detection + CNN prediction."""
    global latest_camera_frame, latest_skeleton_frame, pts, ccc
    global str_output, word, word1, word2, word3, word4
    global current_symbol, prev_char, count, ten_prev_char

    data = request.get_json()
    if not data or 'frame' not in data:
        return jsonify({'ok': False, 'error': 'No frame data'}), 400

    try:
        # Decode base64 frame from browser
        frame_data = data['frame']
        # Strip data URL prefix if present
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]

        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        cv2image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if cv2image is None:
            return jsonify({'ok': False, 'error': 'Failed to decode frame'}), 400

        cv2image = cv2.flip(cv2image, 1)

        with frame_lock:
            latest_camera_frame = cv2image.copy()

        if active_mode == "action":
            # Action Detection Mode
            skeleton_b64 = None
            if 'mp_holistic' not in globals() or 'mp_drawing' not in globals():
                return jsonify({'ok': False, 'error': 'Action detection not fully initialized'})

            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                processed_image, results = mediapipe_detection(cv2image.copy(), holistic)
                draw_action_landmarks(processed_image, results)
                keypoints = extract_action_keypoints(results)
                action_sequence.append(keypoints)
                
                # Ensure sequence is capped
                while len(action_sequence) > ACTION_SEQUENCE_LENGTH:
                    action_sequence.pop(0)

                if action_model is None:
                    action_current_symbol = "Model missing"
                    action_probabilities = [0.0 for _ in action_labels]
                elif len(action_sequence) == ACTION_SEQUENCE_LENGTH:
                    res = action_model.predict(np.expand_dims(action_sequence, axis=0), verbose=0)[0]
                    action_probabilities = res.tolist()
                    predicted_index = int(np.argmax(res))
                    action_current_symbol = action_labels[predicted_index]
                    action_predictions.append(predicted_index)
                    
                    while len(action_predictions) > ACTION_STABILITY_WINDOW:
                        action_predictions.pop(0)

                    stable_predictions = action_predictions[-ACTION_STABILITY_WINDOW:]
                    if (
                        len(stable_predictions) == ACTION_STABILITY_WINDOW
                        and len(set(stable_predictions)) == 1
                        and res[predicted_index] > ACTION_THRESHOLD
                    ):
                        predicted_label = action_labels[predicted_index]
                        if not action_sentence or predicted_label != action_sentence[-1]:
                            action_sentence.append(predicted_label)

                    if len(action_sentence) > 5:
                        action_sentence.pop(0)

                    processed_image = action_prob_viz(res, action_labels, processed_image, action_colors)

                cv2.rectangle(processed_image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(processed_image, ' '.join(action_sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                with frame_lock:
                    latest_skeleton_frame = processed_image.copy()

                _, skel_jpg = cv2.imencode('.jpg', processed_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                skeleton_b64 = base64.b64encode(skel_jpg.tobytes()).decode('utf-8')

            return jsonify({
                'ok': True,
                'character': str(action_current_symbol) if action_current_symbol else '-',
                'sentence': ' '.join(action_sentence),
                'action_probabilities': action_probabilities,
                'skeleton': skeleton_b64
            })
            
        else:
            # Alphabet Detection Mode
            skeleton_b64 = None
            hands = normalize_hands_result(hd.findHands(cv2image, draw=False, flipType=True))
            cv2image_copy = np.array(cv2image)

        if hands:
            hand = hands[0]
            x, y, w_box, h_box = hand['bbox']
            y1 = max(0, y - offset)
            y2 = min(cv2image_copy.shape[0], y + h_box + offset)
            x1 = max(0, x - offset)
            x2 = min(cv2image_copy.shape[1], x + w_box + offset)
            image = cv2image_copy[y1:y2, x1:x2]
            white = np.ones((400, 400, 3), dtype=np.uint8) * 255

            if image is not None and image.size > 0:
                handz = normalize_hands_result(hd.findHands(image, draw=False, flipType=True))
                ccc += 1
                if handz:
                    hand2 = handz[0]
                    pts = hand2['lmList']
                    os_x = ((400 - w_box) // 2) - 15
                    os_y = ((400 - h_box) // 2) - 15

                    for t in range(0, 4):
                        cv2.line(white, (pts[t][0]+os_x, pts[t][1]+os_y), (pts[t+1][0]+os_x, pts[t+1][1]+os_y), (0,255,0), 3)
                    for t in range(5, 8):
                        cv2.line(white, (pts[t][0]+os_x, pts[t][1]+os_y), (pts[t+1][0]+os_x, pts[t+1][1]+os_y), (0,255,0), 3)
                    for t in range(9, 12):
                        cv2.line(white, (pts[t][0]+os_x, pts[t][1]+os_y), (pts[t+1][0]+os_x, pts[t+1][1]+os_y), (0,255,0), 3)
                    for t in range(13, 16):
                        cv2.line(white, (pts[t][0]+os_x, pts[t][1]+os_y), (pts[t+1][0]+os_x, pts[t+1][1]+os_y), (0,255,0), 3)
                    for t in range(17, 20):
                        cv2.line(white, (pts[t][0]+os_x, pts[t][1]+os_y), (pts[t+1][0]+os_x, pts[t+1][1]+os_y), (0,255,0), 3)
                    cv2.line(white, (pts[5][0]+os_x, pts[5][1]+os_y), (pts[9][0]+os_x, pts[9][1]+os_y), (0,255,0), 3)
                    cv2.line(white, (pts[9][0]+os_x, pts[9][1]+os_y), (pts[13][0]+os_x, pts[13][1]+os_y), (0,255,0), 3)
                    cv2.line(white, (pts[13][0]+os_x, pts[13][1]+os_y), (pts[17][0]+os_x, pts[17][1]+os_y), (0,255,0), 3)
                    cv2.line(white, (pts[0][0]+os_x, pts[0][1]+os_y), (pts[5][0]+os_x, pts[5][1]+os_y), (0,255,0), 3)
                    cv2.line(white, (pts[0][0]+os_x, pts[0][1]+os_y), (pts[17][0]+os_x, pts[17][1]+os_y), (0,255,0), 3)

                    for i in range(21):
                        cv2.circle(white, (pts[i][0]+os_x, pts[i][1]+os_y), 2, (0,0,255), 1)

                    predict(white)

                with frame_lock:
                    latest_skeleton_frame = white.copy()

                # Encode skeleton as base64 to send back to browser
                _, skel_jpg = cv2.imencode('.jpg', white, [cv2.IMWRITE_JPEG_QUALITY, 85])
                skeleton_b64 = base64.b64encode(skel_jpg.tobytes()).decode('utf-8')
        else:
            white = np.ones((400, 400, 3), dtype=np.uint8) * 255
            with frame_lock:
                latest_skeleton_frame = white.copy()
            _, skel_jpg = cv2.imencode('.jpg', white, [cv2.IMWRITE_JPEG_QUALITY, 85])
            skeleton_b64 = base64.b64encode(skel_jpg.tobytes()).decode('utf-8')

        return jsonify({
            'ok': True,
            'character': str(current_symbol) if current_symbol else '-',
            'sentence': str_output,
            'word1': word1,
            'word2': word2,
            'word3': word3,
            'word4': word4,
            'skeleton': skeleton_b64
        })

    except Exception as exc:
        print("process-frame error:", traceback.format_exc())
        return jsonify({'ok': False, 'error': str(exc)}), 500



@app.route('/clear', methods=['POST'])
@app.route('/api/clear', methods=['POST'])
@login_required
def clear():
    """Clear — mirrors clear_fun from final_pred.py."""
    global str_output, word1, word2, word3, word4
    global analysis_result, analysis_status
    analysis_result = ""
    analysis_status = "idle"
    if active_mode == "action":
        reset_action_state()
    else:
        str_output = " "
        word1 = word2 = word3 = word4 = " "
    return jsonify({'ok': True})


@app.route('/health')
@login_required
def health():
    return jsonify({'ok': True, 'state': build_state()})


@app.route('/api/cameras')
@login_required
def list_cameras():
    """List all available cameras."""
    global available_cameras
    available_cameras = discover_cameras()
    return jsonify({
        'cameras': available_cameras,
        'current': current_camera_idx
    })


@app.route('/api/camera/switch', methods=['POST'])
@login_required
def camera_switch():
    """Switch to a different camera."""
    data = request.get_json()
    new_idx = data.get('index')
    if new_idx is None:
        return jsonify({'ok': False, 'error': 'Missing camera index'}), 400
    success = switch_camera(int(new_idx))
    if success:
        return jsonify({'ok': True, 'current': current_camera_idx})
    return jsonify({'ok': False, 'error': f'Camera {new_idx} not available'}), 400


@app.route('/api/camera/rotate', methods=['POST'])
@login_required
def camera_rotate():
    """Rotate camera feed. Cycles 0→90→180→270→0, or set specific degrees."""
    global camera_rotation
    data = request.get_json() or {}
    if 'degrees' in data:
        camera_rotation = int(data['degrees']) % 360
    else:
        # Cycle to next rotation
        camera_rotation = (camera_rotation + 90) % 360
    return jsonify({'ok': True, 'rotation': camera_rotation})

# ---------------------------------------------------------------------------
# Start background video thread and run Flask
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if not CLOUD_MODE:
        video_thread = threading.Thread(target=video_loop, daemon=True)
        video_thread.start()
    else:
        print("☁️  Skipping video_loop (browser webcam handles frames)")
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  * Sign Language App running at  http://0.0.0.0:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
