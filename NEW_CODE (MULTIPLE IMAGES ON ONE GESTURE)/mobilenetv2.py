import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Load MobileNetV2 as the base model (pretrained on ImageNet)
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')

# Freeze the base model layers (optional)
base_model.trainable = False  # Change to True for fine-tuning later

# Add new layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduces dimensions
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model WITHOUT Early Stopping
model.fit(train_generator, validation_data=validation_generator, epochs=EPOCHS)

# Save Model
model.save("medical_sign_language_model.h5")
print("✅ Model saved successfully!")
