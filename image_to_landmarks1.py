import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Define paths
dataset_folder = "latest_dataset" 
output_csv = "latest_dataset_landmarks.csv"

data = []


# Process each PNG image
for image_name in sorted(os.listdir(dataset_folder)):
    # Extract label from filename (remove ".png" extension)
    sign_label = os.path.splitext(image_name)[0].split(" ")[0]

    # Load image
    image_path = os.path.join(dataset_folder, image_name) 
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            data.append(landmarks + [sign_label])  # Append gesture label

# Save extracted landmarks to CSV
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False, header=False)
print(f"Extracted landmarks saved to {output_csv}.")