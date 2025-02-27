import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# 📌 Load the custom medical sign language model
model = tf.keras.models.load_model("medical_sign_language_model.h5")

# 📌 Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# 📌 Define Medical Sign Labels (Replace with your actual gesture names)
gesture_labels = ["ALLERGY", "COLD", "COUGH", "DIZZY", "DOCTOR", "FEVER", "GOOD MORNING", "HURT", "NURSE", "SICK", "THANK YOU VERY MUCH", "THANK YOU", "TIRED", "VOMIT"]

# 📌 Start Video Capture
cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 📌 Process the frame with MediaPipe
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 📌 Extract Hand Landmark Coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Convert landmarks to NumPy array and reshape for model input
            landmarks = np.array(landmarks).reshape(1, -1)

            # 📌 Predict Gesture
            prediction = model.predict(landmarks)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100  # Convert to percentage
            sign_text = f"{gesture_labels[predicted_class]} ({confidence:.2f}%)"

            # 📌 Get Position to Overlay Text Above the Hand
            x_pos = int(hand_landmarks.landmark[0].x * w)
            y_pos = int(hand_landmarks.landmark[0].y * h) - 20  # Adjust above the hand

            # 📌 Display Prediction on Video (Above the Hand)
            cv2.putText(frame, sign_text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("Medical Sign Language Detection", frame)

    # Exit on 'q' key press or if window is closed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Medical Sign Language Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()