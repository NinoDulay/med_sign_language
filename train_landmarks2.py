import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("latest_dataset_landmarks.csv", header=None)

# Extract features and labels
X = df.iloc[:, :-1].values  # Hand landmarks
y = df.iloc[:, -1].values   # Gesture labels

# Normalize the features (landmarks)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode labels (Convert gesture labels to numbers)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Save label mappings
np.save("medical_sign_classes.npy", encoder.classes_)

# Define the model with Dropout and Batch Normalization
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.BatchNormalization(),  # Batch normalization
    tf.keras.layers.Dropout(0.5),  # Dropout to prevent overfitting
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

# Compile the model with a different optimizer (RMSprop)
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, batch_size=32, verbose=1)

# Save the trained model
model.save("medical_sign_language_model.h5")
print("Model trained and saved as 'medical_sign_language_model.h5'.")





# import tensorflow as tf
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder

# # Load dataset
# df = pd.read_csv("latest_dataset_landmarks.csv", header=None)

# # Extract features and labels
# X = df.iloc[:, :-1].values  # Hand landmarks
# y = df.iloc[:, -1].values   # Gesture labels

# # Encode labels (Convert gesture labels to numbers)
# encoder = LabelEncoder()
# y = encoder.fit_transform(y)

# # Save label mappings
# np.save("medical_sign_classes.npy", encoder.classes_)

# # Define the model (Simplified architecture)
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),  # Increased layer size
#     tf.keras.layers.Dense(128, activation='relu'),  # Added more neurons
#     tf.keras.layers.Dense(64, activation='relu'),   # Another layer with more neurons
#     tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')  # Output layer with softmax
# ])

# # Compile & Train Model on entire dataset
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(X, y, epochs=100, batch_size=32)  # Increased epochs and batch size for better learning

# # Save trained model
# model.save("medical_sign_language_model.h5")
# print("Model trained on entire dataset and saved as 'medical_sign_language_model.h5'.")


# import tensorflow as tf
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# # Load dataset
# df = pd.read_csv("latest_dataset_landmarks.csv", header=None)

# # Extract features and labels
# X = df.iloc[:, :-1].values  # Hand landmarks
# y = df.iloc[:, -1].values   # Gesture labels

# # Encode labels (Convert words to numbers)
# encoder = LabelEncoder()
# y = encoder.fit_transform(y)

# # Save label mappings
# np.save("medical_sign_classes.npy", encoder.classes_)

# # Split dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# # Define the model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
# ])

# # Compile & Train Model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# # Save trained model
# model.save("medical_sign_language_model.h5")
# print("Model trained and saved as 'medical_sign_language_model.h5'.")