import numpy as np
import matplotlib.pyplot as plt

# Load the history dictionary
history_path = 'exp3/exp_3_(rdi+rai)_rdi_mesca_gmn.npy'
history = np.load(history_path, allow_pickle=True).item()

# Plot and save loss curve
plt.figure()
plt.plot(history['loss'], label='train loss')
if 'val_loss' in history:
    plt.plot(history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('exp3_loss_curve.png')   # ← saved here
plt.close()

# Plot and save accuracy curve (if present)
if 'accuracy' in history:
    plt.figure()
    plt.plot(history['accuracy'], label='train acc')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='val acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('exp3_accuracy_curve.png')   # ← saved here
    plt.close()

print("Saved plots as 'loss_curve.png' and 'accuracy_curve.png'")
