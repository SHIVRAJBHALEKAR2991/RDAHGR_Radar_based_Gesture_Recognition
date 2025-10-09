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

# Load model architecture
with open("exp_results/exp1/exp_1_mesca_early_architecture.json", "r") as json_file:
    model_json = json_file.read()

# model = tf.keras.models.model_from_json(
#     model_json,
#     custom_objects={
#         'ArcFace': ArcFace,
#         'Cross_MSECA_Module': Cross_MSECA_Module,
#         'TEA_MTA': TEA_MTA,
#         'CT_Module': CT_Module,
#         'two_plus_oneDConv': two_plus_oneDConv,
#         'FlattenTemporal': FlattenTemporal,
#         'RestoreShape': RestoreShape,
#         'L2': tf.keras.regularizers.l2,
#         'GlorotUniform': tf.keras.initializers.GlorotUniform,
#         'Zeros': tf.keras.initializers.Zeros,
#         'HeNormal': tf.keras.initializers.HeNormal
#     }
# )
####### Model Training

####### Defining Layers and Model

###### Defining Layers

##### Input Shapes
T = 40
H = 32
W = 32
C_rdi = 4
C_rai = 1

##### Convolutional Layers

#### RDI
conv_up1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), padding='same',
                                  activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
conv11_rdi = CT_Module(40, 32, 32, 32)
conv12_rdi = CT_Module(40, 32, 32, 32)
conv13_rdi = CT_Module(40, 32, 32, 32)

conv_up2 = tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 1), padding='same',
                                  activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
conv21_rdi = CT_Module(40, 32, 32, 64)
conv22_rdi = CT_Module(40, 32, 32, 64)
conv23_rdi = CT_Module(40, 32, 32, 64)

#### RAI
conv11_rai = two_plus_oneDConv(32, 3, 32, 32, 1, 40)
conv12_rai = two_plus_oneDConv(32, 3, 32, 32, 32 + 1, 40)
conv13_rai = two_plus_oneDConv(32, 3, 32, 32, 32 + 32 + 1, 40)

conv21_rai = two_plus_oneDConv(64, 3, 32, 32, 32, 40)
conv22_rai = two_plus_oneDConv(64, 3, 32, 32, 64 + 32, 40)
conv23_rai = two_plus_oneDConv(64, 3, 32, 32, 64 + 64 + 32, 40)

##### Channel Attention Module
# jlce_module = JLCE(1,5,64)
# cam3d = CAM3D(128,40,32,32,1)
# eca_module = ECA_Module(40,32,32,128,1,1)
# optisecam3d_shuffle = OptiSECAM3D_Shuffle(128,1)
cross_mseca_module = Cross_MSECA_Module(40, 32, 32, 5, 3)

##### TEA
#### TEA-1
conv1_TEA1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                                    activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
# TEA_ME_1 = TEA_ME(4, 128)
TEA_MTA_1 = TEA_MTA(2, 40, 32, 32, 128)
conv2_TEA1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                                    activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

#### TEA-2
conv1_TEA2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                                    activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
# TEA_ME_2 = TEA_ME(8, 256)
TEA_MTA_2 = TEA_MTA(2, 40, 32, 32, 128)
conv2_TEA2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                                    activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

#### TEA-3
conv1_TEA3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                                    activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
# TEA_ME_3 = TEA_ME(16, 512)
TEA_MTA_3 = TEA_MTA(2, 40, 32, 32, 128)
conv2_TEA3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same',
                                    activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

##### ArcFace Loss
arc_logit_layer = ArcFace(11, 30.0, 0.2, tf.keras.regularizers.l2(1e-4))

###### Defining Model

##### Input Layer
Input_Layer_rdi = tf.keras.layers.Input(shape=(T, H, W, C_rdi))
Input_Layer_rai = tf.keras.layers.Input(shape=(T,H,W,C_rai))
Input_Labels = tf.keras.layers.Input(shape=(11,))
# Input_Layer_rdi = tf.keras.layers.Input(shape=(None, H, W, C_rdi))
# Input_Layer_rdi = tf.keras.layers.Input(shape=(40, 32, 32, 4))  # Fix temporal dimension
# Input_Labels = tf.keras.layers.Input(shape=(11,))


##### Conv Layers

#### RDI
### Tensorized Residual Block - 1
# print("input_layer_rdi",Input_Layer_rdi.shape)
# conv_up1 = conv_up1(Input_Layer_rdi)
# print("conv_up1 size",conv_up1.shape)
# conv11_rdi = conv11_rdi(conv_up1)
# conv12_rdi = conv12_rdi(conv11_rdi)
# print("Before Add: conv12_rdi", conv12_rdi.shape, "conv_up1", conv_up1.shape)
# conv12_rdi = tf.keras.layers.Add()([conv12_rdi, conv_up1])
# print("After Add: conv12_rdi", conv12_rdi.shape, "conv_up1", conv_up1.shape)
# conv12_rdi = tf.keras.layers.Add()([conv12_rdi, conv_up1])
#
# conv12_rdi = tf.keras.layers.Add()([conv12_rdi, conv_up1])
# conv13_rdi = conv13_rdi(conv12_rdi)
# conv13_rdi = tf.keras.layers.Add()([conv13_rdi, conv11_rdi])

### Tensorized Residual Block - 2
# conv_up2 = conv_up2(conv13_rdi)
# conv21_rdi = conv21_rdi(conv_up2)
# conv22_rdi = conv22_rdi(conv21_rdi)
# conv22_rdi = tf.keras.layers.Add()([conv22_rdi, conv_up2])
# conv23_rdi = conv23_rdi(conv22_rdi)
# conv23_rdi = tf.keras.layers.Add()([conv23_rdi, conv21_rdi])

#### RAI
### Dense Block - 1
# conv11_rai = conv11_rai(Input_Layer_rai)
# conv11_rai = tf.keras.layers.Concatenate(axis=-1)([conv11_rai,Input_Layer_rai])
# conv12_rai = conv12_rai(conv11_rai)
# conv12_rai = tf.keras.layers.Concatenate(axis=-1)([conv12_rai,conv11_rai])
# conv13_rai = conv13_rai(conv12_rai)

### Dense Block - 2
# conv21_rai = conv21_rai(conv13_rai)
# conv21_rai = tf.keras.layers.Concatenate(axis=-1)([conv21_rai,conv13_rai])
# conv22_rai = conv22_rai(conv21_rai)
# conv22_rai = tf.keras.layers.Concatenate(axis=-1)([conv22_rai,conv21_rai])
# conv23_rai = conv23_rai(conv22_rai)

#### Concatenation Operation
conv23 = tf.keras.layers.Concatenate(axis=-1)([Input_Layer_rdi,Input_Layer_rai])

##### Channel Attention
print("entering into the mesca modeule  !!!!!")
print("size entering the mesca module",conv23.shape)
conv23_cross_mseca = cross_mseca_module(conv23)
print("left the cross mesca module !!!!!!!!!!!!!!!!!!!!!!!!")
print("after the mesca modeule ",conv23_cross_mseca.shape)
conv23_cross_mseca = tf.keras.layers.Add()([conv23_cross_mseca, conv23])

# optisecam3d_shuffle_op = optisecam3d_shuffle(conv23)

#### TEA-1
# print(f"conv23_cross_mesra {conv23_cross_mseca.shape}")
def safe_reshape(x, shape):
    # print("Before Reshape:", x.shape)|
    reshaped_x = tf.reshape(x, shape)
    # print("After Reshape:", reshaped_x.shape)
    return reshaped_x

flatten_temporal = FlattenTemporal()
restore_shape = RestoreShape()

# Apply the Conv2D layer
print("conv23_cross_mesca",conv23_cross_mseca.shape)
conv1_tea1 = flatten_temporal(conv23_cross_mseca)  # Flatten
conv1_tea1 = conv1_TEA1(conv1_tea1)
conv1_tea1 = restore_shape(conv1_tea1)  # Restore

tea_mta1 = TEA_MTA_1(conv1_tea1)
reshaped_tea_mta1 = flatten_temporal(tea_mta1)  # Flatten
conv2_tea1_temp = conv2_TEA1(reshaped_tea_mta1)
conv2_tea1 = restore_shape(conv2_tea1_temp)  # Restore
print("conv1_tea1",conv1_tea1.shape,"conv2_tea1",conv2_tea1.shape)
#tea1_op = tf.keras.layers.Add()([conv1_tea1, conv2_tea1])

#### TEA-2
#print("tea1_op",tea1_op.shape)
tea1_op_reshaped = flatten_temporal(conv2_tea1)
conv1_tea2 = conv1_TEA2(tea1_op_reshaped)
conv1_tea2 = restore_shape(conv1_tea2)

tea_mta2 = TEA_MTA_2(conv1_tea2)
tea_mta2 = flatten_temporal(tea_mta2)
conv2_tea2 = conv2_TEA2(tea_mta2)
conv2_tea2 = restore_shape(conv2_tea2)

#print("conv2_tea2",conv2_tea2.shape,"tea1_op",tea1_op.shape)
#tea2_op = tf.keras.layers.Add()([conv2_tea2, tea1_op])

#### TEA-3
tea2_op = flatten_temporal(conv2_tea2)
conv1_tea3 = conv1_TEA3(tea2_op)
conv1_tea3 = restore_shape(conv1_tea3)

tea_mta3 = TEA_MTA_3(conv1_tea3)
tea_mta3 = flatten_temporal(tea_mta3)
conv2_tea3 = conv2_TEA3(tea_mta3)
conv2_tea3 = restore_shape(conv2_tea3)

#tea3_op = tf.keras.layers.Add()([conv2_tea3, tea2_op])

#print(f"GMN BAAD WALI BRANCH {tea3_op.shape}")
#### Output Layer
gap_op = tf.keras.layers.GlobalAveragePooling3D()(conv2_tea3)
dense1 = tf.keras.layers.Dense(256, activation='relu')(gap_op)
dropout1 = tf.keras.layers.Dropout(rate=0.2)(dense1)

### Softmax Output Layer
# dense2 = tf.keras.layers.Dense(256,activation='relu')(dropout1)
# dropout2 = tf.keras.layers.Dropout(rate=0.2)(dense2)
# dense3 = tf.keras.layers.Dense(11,activation='softmax')(dense2)

### ArcFace Output Layer
dense2 = tf.keras.layers.Dense(256, kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dropout1)
##dense2 = tf.keras.layers.BatchNormalization()(dense2)
dense3 = arc_logit_layer(([dense2, Input_Labels]))

###### Compiling Model
model = tf.keras.models.Model(inputs=[Input_Layer_rdi, Input_Layer_rai,Input_Labels], outputs=dense3)
model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# Load weights
model.load_weights("exp_results/exp1/exp_1_mesca_early_weights.h5")

# Run predictions
y_pred_probs = model.predict([X_dev_rdi, X_dev_rai, y_dev_onehot], batch_size=2)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_dev_onehot, axis=1)

# Evaluate per-class accuracy
for i, class_name in enumerate(class_names):
    indices = np.where(y_true == i)[0]
    class_acc = accuracy_score(y_true[indices], y_pred[indices])
    print(f"{class_name}: {class_acc * 100:.2f}%")
