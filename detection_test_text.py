import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk

# 📌 Load the custom medical sign language model
model = tf.keras.models.load_model("medical_sign_language_model.h5")

# 📌 Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# 📌 Define Medical Sign Labels (Replace with your actual gesture names)
gesture_labels = ["ALLERGY", "COLD", "COUGH", "DIZZY", "FEVER", "HURT", "SICK", "TIRED", "VOMIT"]

# 📌 GUI Setup
root = tk.Tk()
root.title("Medical Sign Language Detection")

# Define the left frame (for the translation text)
left_frame = tk.Frame(root, width=200, height=480, bg="white")
left_frame.grid(row=0, column=0, padx=10, pady=10)

# Define the right frame (for the video display)
right_frame = tk.Frame(root, width=640, height=480)
right_frame.grid(row=0, column=1, padx=10, pady=10)

# Translation display
translation_label = tk.Label(left_frame, text="Detected Gesture:\n\n", bg="white", font=("Arial", 16), justify="left")
translation_label.pack(padx=10, pady=10)

# Clear button to reset the translation
def clear_translation():
    translation_label.config(text="Detected Gesture:\n\n")

clear_button = tk.Button(left_frame, text="Clear", command=clear_translation, font=("Arial", 12), bg="red", fg="white")
clear_button.pack(pady=20)

# 📌 Start Video Capture
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if ret:
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

                # Update the translation label
                translation_label.config(text=f"Detected Gesture:\n\n{sign_text}")

        # Convert the frame to PIL format for Tkinter
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Display the video frame in the right_frame
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # Call the function again after 10 milliseconds
    root.after(10, update_frame)

# Create a label in the right_frame for video
video_label = tk.Label(right_frame)
video_label.pack()

# Start the video capture loop
update_frame()

# Run the Tkinter event loop
root.mainloop()

# Release resources when the window is closed
cap.release()
cv2.destroyAllWindows()
