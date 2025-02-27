import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ========== PARAMETERS ==========
DATASET_PATH = "medical_signs_dataset"  # Folder with gesture subfolders
IMG_SIZE = 64  # Image size for training
BATCH_SIZE = 32  # Number of images processed at a time
EPOCHS = 20  # Number of training cycles

# ========== DATA LOADING ==========
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize images
    validation_split=0.2  # 80% training, 20% validation
)

# Training data generator (loads images independently)
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ========== CNN MODEL ==========
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Helps prevent overfitting
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# ========== TRAIN THE MODEL ==========
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# Save the trained model
MODEL_SAVE_PATH = "medical_sign_language_model.h5"
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved as {MODEL_SAVE_PATH}")
