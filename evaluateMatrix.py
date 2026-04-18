import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Set paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(SCRIPT_DIR, "splitDataset", "test") 
MODEL_PATH = os.path.join(SCRIPT_DIR, 'soil_B7_model.keras')

# Parameters (Must match training exactly)
IMG_SIZE = 600
BATCH_SIZE = 4

print("Loading Test Data...")
# CRITICAL: shuffle=False. If we shuffle here, the predictions won't match the true labels!
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False  
)

class_names = test_ds.class_names

print(f"Loading Model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Extracting true labels...")
true_labels = []
for images, labels in test_ds:
    # Convert from one-hot encoding back to integers (0, 1, 2, 3, 4)
    true_labels.extend(np.argmax(labels.numpy(), axis=1))

print("Making predictions on the Test set...")
predictions = model.predict(test_ds)
predicted_labels = np.argmax(predictions, axis=1)

print("Generating Confusion Matrix...")
cm = confusion_matrix(true_labels, predicted_labels)

# Plotting the Confusion Matrix nicely
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted Mixture by Model', fontsize=12, fontweight='bold')
plt.ylabel('Actual Mixture (Ground Truth)', fontsize=12, fontweight='bold')
plt.title(f'EfficientNet-B7 Confusion Matrix (Test Acc: {np.mean(np.array(true_labels) == np.array(predicted_labels)):.2%})', fontsize=14)

plt.tight_layout()

plt.show()