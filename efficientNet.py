import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB7    
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt 
import os
import numpy as np

# Set up the folder paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SCRIPT_DIR, "splitDataset")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val") 
TEST_DIR = os.path.join(DATASET_ROOT, "test") 

# Parameters for the model
IMG_SIZE = 600
BATCH_SIZE = 4 # NOTE B7 is massive; if you get a "Memory Error" (OOM), lower this to 2 or 1.
EPOCHS = 30

# Load the training data
print("Loading Training Data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Load the validation data (for the training loop)
print("Loading Validation Data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical' 
)

# Load the test data (for the final evaluation only)
print("Loading Testing Data...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical' 
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

# Optimise the data pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Build the model with Augmentation Layers inside
def build_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # These only run during training (model.fit)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(factor=0.5, fill_mode='constant', fill_value=255.0)(x) # 0.5 factor allows 90-degree steps
    x = layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant', fill_value=255.0)(x)
    
    base_model = EfficientNetB7(
        include_top=False, 
        weights="imagenet", 
        input_tensor=x
    )
    base_model.trainable = False  # Freeze the base for now

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x) # Slightly higher dropout for B7
    
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x) # not a regression model but returns the probability of the classes out of 1.0

    model = models.Model(inputs, outputs)
    return model

print("Building EfficientNet-B7 Model...")
model = build_model()

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Built in early stopping when the validation loss reaches a point of no return
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True
)

print("Starting Training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# Evaluate the model based on unseen test data 
print("\n Testing this shit...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

# Save model
save_path = os.path.join(SCRIPT_DIR, 'soil_B7_model.keras')
model.save(save_path)
print(f"Model saved to {save_path}")
