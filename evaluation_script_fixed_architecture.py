import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Import custom layers from training script
from exp_1_mesca_early_fusion import (
    ArcFace, Cross_MSECA_Module, TEA_MTA, CT_Module, two_plus_oneDConv,
    FlattenTemporal, RestoreShape
)

# Define gesture class names
class_names = [
    'Pinch index', 'Palm tilt', 'Finger slide', 'Pinch pinky', 'Slow swipe',
    'Fast swipe', 'Push', 'Pull', 'Finger rub', 'Circle', 'Palm hold'
]

# Load validation dataset
X_dev_rdi = np.load('Dataset/X_dev_rdi_soli.npz', allow_pickle=True)['arr_0']
X_dev_rai = np.load('Dataset/X_dev_rai_soli.npz', allow_pickle=True)['arr_0']
y_dev = np.load('Dataset/y_dev_soli.npz', allow_pickle=True)['arr_0']
y_dev_onehot = tf.keras.utils.to_categorical(y_dev, num_classes=len(class_names))

# Load the FIXED model architecture
print("Loading fixed model architecture...")
with open("exp_1_mesca_early_architecture_fixed.json", "r") as json_file:
    model_json = json_file.read()

# Load model with all custom objects
model = tf.keras.models.model_from_json(
    model_json,
    custom_objects={
        'ArcFace': ArcFace,
        'Cross_MSECA_Module': Cross_MSECA_Module,
        'TEA_MTA': TEA_MTA,
        'CT_Module': CT_Module,
        'two_plus_oneDConv': two_plus_oneDConv,
        'FlattenTemporal': FlattenTemporal,
        'RestoreShape': RestoreShape,
        'L2': tf.keras.regularizers.l2,
        'GlorotUniform': tf.keras.initializers.GlorotUniform,
        'Zeros': tf.keras.initializers.Zeros,
        'HeNormal': tf.keras.initializers.HeNormal
    }
)

print("Model architecture loaded successfully!")
print("Model summary:")
model.summary()

# Load weights
print("Loading weights...")
model.load_weights("exp_1_mesca_early_weights_fixed.h5")
print("Weights loaded successfully!")

# Compile the model (needed after loading from JSON)
model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Run predictions
print("Running predictions...")
y_pred_probs = model.predict([X_dev_rdi, X_dev_rai, y_dev_onehot], batch_size=2)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_dev_onehot, axis=1)

# Calculate overall accuracy
overall_acc = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {overall_acc * 100:.2f}%")

# Evaluate per-class accuracy
print("\nPer-class Accuracy:")
for i, class_name in enumerate(class_names):
    indices = np.where(y_true == i)[0]
    if len(indices) > 0:
        class_acc = accuracy_score(y_true[indices], y_pred[indices])
        print(f"{class_name}: {class_acc * 100:.2f}%")
    else:
        print(f"{class_name}: No samples in validation set")

print("\nEvaluation completed successfully!") 