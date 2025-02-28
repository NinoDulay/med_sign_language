import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import os

# Fixed overfitting

# ========== PARAMETERS ==========
DATASET_PATH = "medical_signs_dataset"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50  # Start with a higher number, early stopping will prevent overtraining

# ========== DATA AUGMENTATION ==========
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2,
    rotation_range=30,  # Randomly rotate images
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest'  # Fill missing pixels
)

# Training Data Generator
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Validation Data Generator (No Augmentation)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
validation_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ========== CNN MODEL WITH OVERFITTING FIX ==========
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.5),  # Dropout for generalization
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ========== EARLY STOPPING ==========
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Print Model Summary
model.summary()

# ========== TRAIN THE MODEL ==========
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# Save the trained model
MODEL_SAVE_PATH = "medical_sign_language_model.h5"
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved as {MODEL_SAVE_PATH}")
