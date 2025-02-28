import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

# Load the trained model
MODEL_PATH = "medical_sign_language_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load gesture class names from the dataset folder names
GESTURE_CLASSES = sorted([f for f in os.listdir("medical_signs_dataset/") 
                          if os.path.isdir(os.path.join("medical_signs_dataset", f))])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Open webcam
cap = cv2.VideoCapture(2)

# Define display settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
THRESHOLD = 0.6  # Confidence threshold
IMG_SIZE = 64  # Must match training size
WINDOW_NAME = "Hand Gesture Recognition"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame for mirror effect and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe to detect hand landmarks
    results = hands.process(rgb_frame)
    hand_position = None  # For placing the label

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Compute bounding box for the hand to position the text
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, y_min = int(min(x_coords) * w), int(min(y_coords) * h)
            # Position the label above the hand
            hand_position = (x_min, y_min - 10)

    # Prepare the frame for model prediction (single image classification)
    input_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    input_frame = input_frame / 255.0  # Normalize
    input_data = np.expand_dims(input_frame, axis=0)  # Shape: (1, IMG_SIZE, IMG_SIZE, 3)

    # Predict gesture for this frame
    predictions = model.predict(input_data)[0]
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]

    if confidence > THRESHOLD:
        sign_name = GESTURE_CLASSES[predicted_index]
        accuracy_text = f"{sign_name}: {confidence:.2%}"

        # Display label near the detected hand (or fixed position if hand not detected)
        label_position = hand_position if hand_position else (20, 50)
        cv2.putText(frame, accuracy_text, label_position, FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the output
    cv2.imshow(WINDOW_NAME, frame)

    # Exit on 'q' key press or when the window is closed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
