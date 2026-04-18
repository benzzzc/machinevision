import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt 
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SCRIPT_DIR, "splitDataset")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val") 
TEST_DIR = os.path.join(DATASET_ROOT, "test") 
OLD_MODEL_PATH = os.path.join(SCRIPT_DIR, 'soil_B7_model.keras')

IMG_SIZE = 600
BATCH_SIZE = 4 

print("Loading Data...")
train_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='categorical')
val_ds = tf.keras.utils.image_dataset_from_directory(VAL_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='categorical')
test_ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='categorical', shuffle=False)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Fine tuning the model

print(f"Loading Phase 1 Model from {OLD_MODEL_PATH}...")
model = tf.keras.models.load_model(OLD_MODEL_PATH)

print("Unfreezing EfficientNet-B7 layers...")
model.trainable = True # note the change here

# Freeze the bottom and mid layers, only want to alter the weights in the top layer
for layer in model.layers[:100]:
    layer.trainable = False

print("Recompiling...")
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5), # NOTE the tiny learning rate we do not want to run into overfitting too early
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Reduces the learning rate by half when the val_loss metric decreases, patienience is the number of epochs it accepts subsequent decreases to the learning rate
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,    
    patience=2,    
    min_lr=1e-7    
)

# Another call back function that stops the training when convergence is met.
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

print("\n Fine Tuning...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[early_stop, lr_scheduler]
)

print("\n Evaluating...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Final Fine-Tuned Test Accuracy: {test_acc:.4f}")

new_save_path = os.path.join(SCRIPT_DIR, 'soil_B7_model_FINETUNED.keras')
model.save(new_save_path)
print(f"Masterpiece saved to {new_save_path}")

print("Generating training history plot...")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))

# Subplot 1: Accuracy
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Fine-Tuning: Training and Validation Accuracy')

# Subplot 2: Loss
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Fine-Tuning: Training and Validation Loss')
plt.xlabel('epoch')

plt.tight_layout()
plt.show()