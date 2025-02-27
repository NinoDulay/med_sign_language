import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("medical_gesture_landmarks.csv", header=None)

# Extract features and labels
X = df.iloc[:, :-1].values  # Hand landmarks
y = df.iloc[:, -1].values   # Gesture labels

# Encode labels (Convert words to numbers)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Save label mappings
np.save("medical_sign_classes.npy", encoder.classes_)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

# Compile & Train Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save trained model
model.save("medical_sign_language_model.h5")
print("Model trained and saved as 'medical_sign_language_model.h5'.")