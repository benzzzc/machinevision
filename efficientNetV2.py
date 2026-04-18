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
BATCH_SIZE = 4 
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

# Load the validation data
print("Loading Validation Data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Load the test data
print("Loading Test Data...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"Detected classes: {class_names}")

# Assuming your folders are sorted alphabetically (0.00, 0.25, 0.50, 0.75, 1.00)
class_weights_dict = {
    0: 1.0,  # sand_0.00 (Standard penalty)
    1: 2.0,  # sand_0.25 (2x penalty)
    2: 3.0,  # sand_0.50 (Hardest, 3x penalty)
    3: 2.0,  # sand_0.75 (2x penalty)
    4: 1.0   # sand_1.00 (Standard penalty)
}
print(f"Applying Class Weights: {class_weights_dict}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Building EfficientNet-B7...")
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Your existing, controlled augmentations
x = layers.RandomFlip("horizontal_and_vertical")(inputs)
x = layers.RandomRotation(0.2)(x)
x = layers.RandomTranslation(0.1, 0.1)(x)

base_model = EfficientNetB7(
    include_top=False, 
    weights="imagenet", 
    input_tensor=x
)

base_model.trainable = False 

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x) 
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001), # Dropped from 0.001 to smooth learning, hopefully to address the noise in the validation
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True
)

print("Starting Training...")
# added targeted class weights 
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop],
    class_weight=class_weights_dict
)

print("\n Testing this shit...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

save_path = os.path.join(SCRIPT_DIR, 'soil_B7_modelV2.keras')
model.save(save_path)
print(f"Model saved to {save_path}")

print("Generating training history plot...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.tight_layout()
plt.show()