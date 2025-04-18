import tensorflow as tf
import numpy as np

# Paths to the saved model files
architecture_path = "exp_1_mesca_early_architecture.json"
weights_path = "exp_1_mesca_early_weights.h5"
history_path = "exp_1_mesca_early_history.npy"

# Load model architecture
with open(architecture_path, 'r') as json_file:
    model_json = json_file.read()

model = tf.keras.models.model_from_json(model_json)

# Load model weights
model.load_weights(weights_path)

# Compile the model (use the same loss and metrics as during training)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(),  # change if needed
    metrics=['accuracy']  # add other metrics as required
)

# Load training history (optional)
history = np.load(history_path, allow_pickle=True).item()

# Now your model is ready for evaluation
# Example:
# test_loss, test_accuracy = model.evaluate(x_test, y_test)

print(" Model loaded and compiled successfully.")
