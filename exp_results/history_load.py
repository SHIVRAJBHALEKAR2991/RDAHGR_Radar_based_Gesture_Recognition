import numpy as np
import matplotlib.pyplot as plt

# Load the history dictionary
history_path = 'exp7/exp_7_hsi_replace_mesca_history.npy'
history = np.load(history_path, allow_pickle=True).item()

# Find the epoch with the best validation accuracy
best_val_acc = max(history['val_accuracy'])
best_epoch = history['val_accuracy'].index(best_val_acc)

# Print the corresponding values
print("=== Best Validation Accuracy ===")
print(f"Epoch         : {best_epoch + 1}")
print(f"Val Accuracy  : {history['val_accuracy'][best_epoch]:.4f}")
print(f"Train Accuracy: {history['accuracy'][best_epoch]:.4f}")
print(f"Val Loss      : {history['val_loss'][best_epoch]:.4f}")
print(f"Train Loss    : {history['loss'][best_epoch]:.4f}")

# Plotting
epochs = range(1, len(history['accuracy']) + 1)

# Accuracy plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, history['accuracy'], label='Train Accuracy')
plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('exp-7_accuracy_plot.png')  # Save the figure
plt.close()

# Loss plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, history['loss'], label='Train Loss')
plt.plot(epochs, history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('exp_7_loss_plot.png')  # Save the figure
plt.close()
