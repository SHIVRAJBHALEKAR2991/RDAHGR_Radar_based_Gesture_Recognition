import numpy as np

# Load the history dictionary
history_path = 'exp6/exp_6_IAFF_GMN_history.npy'
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