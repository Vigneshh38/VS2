"""
Action Detection for Sign Language
====================================
Run this script to:
  1. Collect training data (webcam)
  2. Train the LSTM model
  3. Run real-time sign language detection

Requirements:
    pip install tensorflow==2.13.0 opencv-python mediapipe==0.10.9 scikit-learn matplotlib scipy
"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from scipy import stats

# ─────────────────────────────────────────────
# 1. MEDIAPIPE SETUP
# ─────────────────────────────────────────────

mp_holistic = mp.solutions.holistic       # Holistic model
mp_drawing  = mp.solutions.drawing_utils  # Drawing utilities
mp_face_mesh = mp.solutions.face_mesh     # Fix: use face_mesh for FACEMESH_CONTOURS


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    # Fix: mp_holistic.FACE_CONNECTIONS removed in newer mediapipe → use FACEMESH_CONTOURS
    mp_drawing.draw_landmarks(image, results.face_landmarks,
                              mp_face_mesh.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10),  thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )
    # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10),  thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )
    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76),  thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    # Draw right hand connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )


# ─────────────────────────────────────────────
# 2. KEYPOINT EXTRACTION
# ─────────────────────────────────────────────

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 4)

    face = np.array([[res.x, res.y, res.z]
                     for res in results.face_landmarks.landmark]).flatten() \
           if results.face_landmarks else np.zeros(468 * 3)

    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])


# ─────────────────────────────────────────────
# 3. CONFIGURATION
# ─────────────────────────────────────────────

DATA_PATH       = os.path.join('MP_Data')
actions         = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences    = 30      # Number of video sequences per action
sequence_length = 30      # Frames per sequence
start_folder    = 30      # Starting folder number


# ─────────────────────────────────────────────
# 4. CREATE FOLDER STRUCTURE
# ─────────────────────────────────────────────

def create_folders():
    for action in actions:
        for sequence in range(1, no_sequences + 1):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except FileExistsError:
                pass
    print("Folders created.")


# ─────────────────────────────────────────────
# 5. COLLECT TRAINING DATA
# ─────────────────────────────────────────────

def collect_data():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(start_folder, start_folder + no_sequences):
                for frame_num in range(sequence_length):

                    ret, frame = cap.read()
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image,
                                    f'Collecting frames for {action} Video Number {sequence}',
                                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(500)
                    else:
                        cv2.putText(image,
                                    f'Collecting frames for {action} Video Number {sequence}',
                                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete.")


# ─────────────────────────────────────────────
# 6. PREPROCESS DATA
# ─────────────────────────────────────────────

def load_data():
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    print(f"Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 7. BUILD & TRAIN MODEL
# ─────────────────────────────────────────────

def build_model():
    model = Sequential()
    model.add(LSTM(64,  return_sequences=True,  activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True,  activation='relu'))
    model.add(LSTM(64,  return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def train_model(X_train, y_train):
    log_dir    = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    model = build_model()
    model.summary()
    model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
    model.save('action.h5')
    print("Model saved to action.h5")
    return model


# ─────────────────────────────────────────────
# 8. EVALUATE MODEL
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    yhat  = model.predict(X_test)
    ytrue = np.argmax(y_test,  axis=1).tolist()
    yhat  = np.argmax(yhat,    axis=1).tolist()
    print("Confusion Matrix:\n", multilabel_confusion_matrix(ytrue, yhat))
    print("Accuracy:", accuracy_score(ytrue, yhat))


# ─────────────────────────────────────────────
# 9. REAL-TIME DETECTION
# ─────────────────────────────────────────────

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40),
                      (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame


def realtime_detection(model):
    sequence    = []
    sentence    = []
    predictions = []
    threshold   = 0.5

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# MAIN — choose which step to run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("""
    Choose a step to run:
      1 - Create folders
      2 - Collect training data (webcam)
      3 - Train model
      4 - Evaluate model
      5 - Real-time detection (load saved model)
      6 - Full pipeline (3+4+5)
    """)
    choice = input("Enter choice (1-6): ").strip()

    if choice == '1':
        create_folders()

    elif choice == '2':
        create_folders()
        collect_data()

    elif choice == '3':
        X_train, X_test, y_train, y_test = load_data()
        train_model(X_train, y_train)

    elif choice == '4':
        from tensorflow.keras.models import load_model
        model = load_model('action.h5')
        _, X_test, _, y_test = load_data()
        evaluate_model(model, X_test, y_test)

    elif choice == '5':
        from tensorflow.keras.models import load_model
        model = load_model('action.h5')
        realtime_detection(model)

    elif choice == '6':
        X_train, X_test, y_train, y_test = load_data()
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        realtime_detection(model)

    else:
        print("Invalid choice.")