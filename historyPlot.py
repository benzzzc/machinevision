import matplotlib.pyplot as plt

# The hardcoded data extracted directly from your terminal output
acc = [0.5549, 0.6730, 0.7172, 0.7637, 0.7506, 0.7721, 0.7936, 0.7959, 0.7900, 0.7900, 
       0.8031, 0.7828, 0.8174, 0.7816, 0.8150, 0.8210, 0.8138, 0.7876, 0.7995, 0.7995]

loss = [1.2869, 0.8438, 0.8284, 0.7604, 0.7773, 0.7023, 0.6072, 0.6402, 0.6592, 0.6412, 
        0.5766, 0.6543, 0.5787, 0.6420, 0.6396, 0.6433, 0.6315, 0.7348, 0.6978, 0.6754]

val_acc = [0.7327, 0.6832, 0.9208, 0.7723, 0.8119, 0.8218, 0.7822, 0.8713, 0.9307, 0.5842, 
           0.5743, 0.7228, 0.9307, 0.8020, 0.7129, 0.7624, 0.8416, 0.8713, 0.8020, 0.6931]

val_loss = [0.7546, 0.7772, 0.2419, 0.6723, 0.6418, 0.5187, 0.8306, 0.4364, 0.1953, 2.4696, 
            2.3689, 1.0100, 0.1280, 0.6557, 0.9613, 1.0406, 0.4277, 0.3292, 1.1511, 1.8112]

epochs_range = range(1, 21) # Epochs 1 to 20

# Create the plot
plt.figure(figsize=(10, 8))

# Subplot 1: Accuracy
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0.4, 1.0]) # Adjusted to fit your data range
plt.title('Training and Validation Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Loss
plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss', color='orange')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()