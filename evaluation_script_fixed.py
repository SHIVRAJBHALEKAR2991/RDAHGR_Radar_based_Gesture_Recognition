import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Import custom layers from training script
from exp_1_mesca_early_fusion import (
    ArcFace, Cross_MSECA_Module, TEA_MTA, CT_Module, two_plus_oneDConv,
    TEA_ME
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

# Rebuild the model from scratch instead of loading JSON
def build_model():
    # Input Shapes
    T = 40
    H = 32
    W = 32
    C_rdi = 4
    C_rai = 1
    
    # Input Layers
    Input_Layer_rdi = tf.keras.layers.Input(shape=(T, H, W, C_rdi))
    Input_Layer_rai = tf.keras.layers.Input(shape=(T, H, W, C_rai))
    Input_Labels = tf.keras.layers.Input(shape=(11,))
    
    # Concatenation Operation
    conv23 = tf.keras.layers.Concatenate(axis=-1)([Input_Layer_rdi, Input_Layer_rai])
    
    # Channel Attention
    cross_mseca_module = Cross_MSECA_Module(40, 32, 32, 5, 3)
    conv23_cross_mseca = cross_mseca_module(conv23)
    conv23_cross_mseca = tf.keras.layers.Add()([conv23_cross_mseca, conv23])
    
    # TEA-1
    flatten_temporal = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, x.shape[2], x.shape[3], x.shape[4])))
    restore_shape = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 40, x.shape[1], x.shape[2], x.shape[3])))
    
    conv1_tea1 = flatten_temporal(conv23_cross_mseca)
    conv1_tea1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                                        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(conv1_tea1)
    conv1_tea1 = restore_shape(conv1_tea1)
    
    tea_mta1 = TEA_MTA(2, 40, 32, 32, 128)(conv1_tea1)
    reshaped_tea_mta1 = flatten_temporal(tea_mta1)
    conv2_tea1_temp = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                                             activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(reshaped_tea_mta1)
    conv2_tea1 = restore_shape(conv2_tea1_temp)
    
    # TEA-2
    tea1_op_reshaped = flatten_temporal(conv2_tea1)
    conv1_tea2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                                        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(tea1_op_reshaped)
    conv1_tea2 = restore_shape(conv1_tea2)
    
    tea_mta2 = TEA_MTA(2, 40, 32, 32, 128)(conv1_tea2)
    tea_mta2 = flatten_temporal(tea_mta2)
    conv2_tea2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                                        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(tea_mta2)
    conv2_tea2 = restore_shape(conv2_tea2)
    
    # TEA-3
    tea2_op = flatten_temporal(conv2_tea2)
    conv1_tea3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                                        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(tea2_op)
    conv1_tea3 = restore_shape(conv1_tea3)
    
    tea_mta3 = TEA_MTA(2, 40, 32, 32, 128)(conv1_tea3)
    tea_mta3 = flatten_temporal(tea_mta3)
    conv2_tea3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                                        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(tea_mta3)
    conv2_tea3 = restore_shape(conv2_tea3)
    
    # Output Layer
    gap_op = tf.keras.layers.GlobalAveragePooling3D()(conv2_tea3)
    dense1 = tf.keras.layers.Dense(256, activation='relu')(gap_op)
    dropout1 = tf.keras.layers.Dropout(rate=0.2)(dense1)
    
    # ArcFace Output Layer
    dense2 = tf.keras.layers.Dense(256, kernel_initializer='he_normal',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dropout1)
    
    arc_logit_layer = ArcFace(11, 30.0, 0.2, tf.keras.regularizers.l2(1e-4))
    dense3 = arc_logit_layer(([dense2, Input_Labels]))
    
    # Create model
    model = tf.keras.models.Model(inputs=[Input_Layer_rdi, Input_Layer_rai, Input_Labels], outputs=dense3)
    return model

# Build the model
print("Building model...")
model = build_model()
model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

print("Model built successfully!")
print("Model summary:")
model.summary()

# Load weights
print("Loading weights...")
model.load_weights("exp_results/exp1/exp_1_mesca_early_weights.h5")
print("Weights loaded successfully!")

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