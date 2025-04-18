import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Number of GPUs detected: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"→ GPU {i}: {gpu}")
else:
    print("No GPU found. Running on CPU.")
