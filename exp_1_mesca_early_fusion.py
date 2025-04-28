# ======================== EARLY FUSION ARCHITECTURE ==========================
#
#                ┌──────────────┐     ┌──────────────┐
#                │   Input RDI  │     │   Input RAI  │
#                │ (T,H,W, C=4) │     │ (T,H,W, C=1) │
#                └──────┬───────┘     └──────┬───────┘
#                       │                   │
#                       └────────┬──────────┘
#                                ▼
#                tf.keras.layers.Concatenate(axis=-1)
#                                ▼
#              ┌─────────────────────────────────────┐
#              │   Fused Input: (T, H, W, C=5)       │
#              └─────────────────────────────────────┘
#                                ▼
#                 ┌────────────────────────────────┐
#                 │         MSECA MODULE           │
#                 │ ────────────────────────────── │
#                 │ • Cross_MSECA_Module()         │
#                 │ • Multi-scale channel attention│
#                 └────────────────────────────────┘
#                                ▼
#                 ┌────────────────────────────────┐
#                 │          GMN MODULE            │
#                 │ (Global Motion Network - prev. │
#                 │      called TEA module)        │
#                 │ ────────────────────────────── │
#                 │ • GMN Layer 1                  │
#                 │ • GMN Layer 2                  │
#                 │ • GMN Layer 3                  │
#                 │   (with MTA attention)         │
#                 └────────────────────────────────┘
#                                ▼
#                 Global Average Pooling (3D)
#                                ▼
#                     Dense Layers + Dropout
#                                ▼
#                     ┌────────────────────┐
#                     │     ArcFace Loss   │
#                     │ (with label input) │
#                     └────────────────────┘
#                                ▼
#                       Final Class Prediction
#
# ============================================================================

####### Importing Libraries
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
#
# plt.switch_backend('agg')
import tensorflow as tf
import os
import gc
import math
# import pydot
# from sklearn.utils import shuffle

####### Loading Dataset

####### Loading Dataset

####### Loading Dataset

###### RDI Sequence
X_train_rdi = np.load('Dataset/X_train_rdi_soli.npz', allow_pickle=True)['arr_0']
X_dev_rdi = np.load('Dataset/X_dev_rdi_soli.npz', allow_pickle=True)['arr_0']

###### RAI Sequence
X_train_rai = np.load('Dataset/X_train_rai_soli.npz', allow_pickle=True)['arr_0']
X_dev_rai = np.load('Dataset/X_dev_rai_soli.npz', allow_pickle=True)['arr_0']
y_train = np.load('Dataset/y_train_soli.npz', allow_pickle=True)['arr_0']
y_dev = np.load('Dataset/y_dev_soli.npz', allow_pickle=True)['arr_0']
###### Converting Labels to Categorical Format
y_train_onehot = tf.keras.utils.to_categorical(y_train)
y_dev_onehot = tf.keras.utils.to_categorical(y_dev)


####### Model Makingclenv

####### TEA Module
####### TEA - Temporal Excitation and Aggregation Network

###### Motion Excitation (ME) Module

class TEA_ME(tf.keras.layers.Layer):
    """ TEA Module's Motion Excitation Block for Motion Modelling """

    def __init__(self, reduction_factor, num_channels):
        #### Defining Essentials
        super().__init__()
        self.reduction_factor = reduction_factor  # Reduction Factor for Reducing Conv
        self.num_channels = num_channels  # Number of Channels in the Input

        #### Defining Layers
        red_val = int(self.num_channels // self.reduction_factor)
        self.conv_red = tf.keras.layers.Conv2D(filters=red_val, kernel_size=(1, 1), padding='same',
                                               activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

        self.conv_transform = tf.keras.layers.Conv2D(filters=red_val, kernel_size=(3, 3), padding='same',
                                                     groups=red_val, activation='relu',
                                                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))

        self.conv_exp = tf.keras.layers.Conv2D(filters=self.num_channels, kernel_size=(1, 1), padding='same',
                                               activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'reduction_factor': self.reduction_factor,
            'num_channels': self.num_channels
        })
        return config

    def call(self, X):
        """
        Implementation of ME Module

        INPUTS:-
        1) X : Input Tensor of Shape [N,T,H,W,C] (Implementation involves 'Channel Last' Strategy)

        OUTPUTS:-
        1) X_o : Tensor of shape [N,T,H,W,C]

        """

        #### Extracting Input Dimensions
        N = (X.shape[0])  # Batch Size
        T = (X.shape[1])  # Total Frames in the Signal
        H = (X.shape[2])  # Height of the Frame
        W = (X.shape[3])  # Width of the Frame
        C = self.num_channels  # Number of Channels in the Input

        #### Reduction of Channel Dimensions
        X_red = self.conv_red(X)

        #### Motion Modelling
        X_red_M1 = X_red[:, :-1, :, :, :]  # Taking the X_red till the penultimate frame
        X_red_M2 = X_red[:, 1:, :, :, :]  # Taking the X_red from the second frame till the end

        X_transform = self.conv_transform(X_red_M2)  # Channel-Wise Convolution

        M = tf.keras.layers.Add()([X_transform, -X_red_M1])  # Action Modelling
        M = tf.keras.layers.ZeroPadding3D(((1, 0), (0, 0), (0, 0)))(M)  # Adding M(T) = 0 Frame

        #### Global Average Pooling
        Ms = tf.keras.layers.AveragePooling3D(pool_size=(1, H, W))(M)

        #### Convolution for Channel Expansion
        Ms_expanded = self.conv_exp(Ms)

        #### Motion Attentive Weights Computation
        A = 2 * (tf.keras.activations.sigmoid(Ms_expanded)) - 1

        #### Output Creation
        X_bar = tf.math.multiply(X, A)  # Channel-wise multiplication of attentive weights
        X_o = tf.keras.layers.Add()([X, X_bar])  # Residual Connection

        return X_o


###### Multiple Temporal Aggregation (MTA) Module

class TEA_MTA(tf.keras.layers.Layer):
    def __init__(self, N, T, H, W, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.N = N
        self.T = T
        self.H = H
        self.W = W
        split_factor = self.num_channels // 4

        self.conv_temp_1 = tf.keras.layers.Conv1D(filters=split_factor, kernel_size=3, padding='same',
                                                  groups=split_factor, activation='relu',
                                                  kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.conv_spa_1 = tf.keras.layers.Conv2D(filters=split_factor, kernel_size=(3, 3), padding='same',
                                                 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

        self.conv_temp_2 = tf.keras.layers.Conv1D(filters=split_factor, kernel_size=3, padding='same',
                                                  groups=split_factor, activation='relu',
                                                  kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.conv_spa_2 = tf.keras.layers.Conv2D(filters=split_factor, kernel_size=(3, 3), padding='same',
                                                 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

        self.conv_temp_3 = tf.keras.layers.Conv1D(filters=split_factor, kernel_size=3, padding='same',
                                                  groups=split_factor, activation='relu',
                                                  kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.conv_spa_3 = tf.keras.layers.Conv2D(filters=split_factor, kernel_size=(3, 3), padding='same',
                                                 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_channels': self.num_channels,
            'N': self.N,
            'T': self.T,
            'H': self.H,
            'W': self.W
        })
        return config

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        temporal_dim = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        channels = input_shape[4]

        output_height = height
        output_width = width
        output_channels = self.num_channels

        return (batch_size, temporal_dim, output_height, output_width, output_channels)

    def call(self, X):
        batch_size = tf.shape(X)[0]  # Dynamically compute batch size
        T = X.shape[1]
        H = X.shape[2]
        W = X.shape[3]
        C = X.shape[4]
        split_factor = C // 4

        Xi_0, Xi_1, Xi_2, Xi_3 = tf.split(X, num_or_size_splits=4, axis=-1)

        Xo_0 = Xi_0

        Xi_1 = tf.keras.layers.Add()([Xo_0, Xi_1])
        Xi_1_reshaped_temp = tf.reshape(Xi_1, [batch_size * T, H * W, split_factor])  #  FIXED
        Xi_1_temp = self.conv_temp_1(Xi_1_reshaped_temp)
        Xi_1_reshaped_spa = tf.reshape(Xi_1_temp, [batch_size, T, H, W, split_factor])  # FIXED
        Xo_1 = self.conv_spa_1(Xi_1_reshaped_spa)

        Xi_2 = tf.keras.layers.Add()([Xo_1, Xi_2])
        Xi_2_reshaped_temp = tf.reshape(Xi_2, [batch_size * T, H * W, split_factor])  # FIXED
        Xi_2_temp = self.conv_temp_2(Xi_2_reshaped_temp)
        Xi_2_reshaped_spa = tf.reshape(Xi_2_temp, [batch_size, T, H, W, split_factor])  # FIXED
        Xo_2 = self.conv_spa_2(Xi_2_reshaped_spa)

        Xi_3 = tf.keras.layers.Add()([Xo_2, Xi_3])
        Xi_3_reshaped_temp = tf.reshape(Xi_3, [batch_size * T, H * W, split_factor])  # FIXED
        Xi_3_temp = self.conv_temp_3(Xi_3_reshaped_temp)
        Xi_3_reshaped_spa = tf.reshape(Xi_3_temp, [batch_size, T, H, W, split_factor])  # FIXED
        Xo_3 = self.conv_spa_3(Xi_3_reshaped_spa)

        Xo = tf.keras.layers.Concatenate(axis=-1)([Xo_0, Xo_1, Xo_2, Xo_3])

        return Xo

####### CT-Module

####### Channel Tensorization (CT-Module)

class CT_Module(tf.keras.layers.Layer):
    """ 3D Tensor Separable Convolution """

    def __init__(self, T, H, W, C):
        ##### Defining Instatiations
        super().__init__()
        self.T = T  # Total number of Frames
        self.H = H  # Height of the Input
        self.W = W  # Width of the Input
        self.C = C  # Channels in the Input

        K = int(math.log2(C))
        k1_dim = int(K / 2)
        self.k1 = int(2 ** (k1_dim))  # Sub Dimension 1
        self.k2 = int(2 ** (K - k1_dim))  # Sub Dimension 2

        ##### Defining Layers
        self.conv_k1 = tf.keras.layers.Conv3D(filters=self.C, kernel_size=(3, 3, 3), padding='same',
                                              activation='linear', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.conv_k2 = tf.keras.layers.Conv3D(filters=self.C, kernel_size=(3, 3, 3), padding='same',
                                              activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'T': self.T,
            'H': self.H,
            'W': self.W,
            'C': self.C,
            'k1': self.k1,
            'k2': self.k2
        })
        return config

    def call(self, X0):
        """
        Implementation of Tensorization Module

        INPUTS:
        1) X0 : Input Tensor of shape (T, H, W, C)

        OUTPUTS:
        1) X2 : Output Tensor of shape (T, H, W, C)
        """
        # Validate input shape
        if X0.shape[-1] != self.C:
            raise ValueError(f"Input channels ({X0.shape[-1]}) do not match expected channels ({self.C}).")

        # Reshape input to split channels
        X0 = tf.keras.layers.Reshape((self.T, self.H, self.W, self.k1 * self.k2))(X0)

        # First Sub-Convolution
        X1 = self.conv_k1(X0)  # Conv3D expects 5D tensor

        # Second Sub-Convolution
        X2 = self.conv_k2(X1)

        # Reshape back to original input dimensions
        X2 = tf.keras.layers.Reshape((self.T, self.H, self.W, self.C))(X2)

        return X2


####### (2+1)D Convolution

####### (2+1)D Convolutional Layer

class two_plus_oneDConv(tf.keras.layers.Layer):
    """ Implementation of(2+1)D Conv """

    def __init__(self, filters, kernel_dims, H, W, C, T):
        #### Defining Essentials
        super().__init__()
        self.filters = filters  # Number of Filters in the Output
        self.kernel_dims = kernel_dims  # Dimensions of the Kernel
        self.H = H  # Height of the Input
        self.W = W  # Width of the Input
        self.C = C  # Number of Channels in the Input
        self.T = T  # Number of Frames in the Input

        #### Defining Layers
        self.conv2d_depthwise = tf.keras.layers.Conv2D(filters=self.C, kernel_size=(self.kernel_dims, self.kernel_dims),
                                                       padding='same', activation='linear', groups=self.C,
                                                       kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.conv2d_pointwise = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1),
                                                       padding='same', activation='relu',
                                                       kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.conv1d = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_dims, padding='same',
                                             activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_dims': self.kernel_dims,
            'H': self.H,
            'W': self.W,
            'C': self.C,
            'T': self.T
        })
        return config

    def call(self, X):
        """
        Implementation of (2+1)D Convolution

        INPUTS:-
        1) X : Input Tensor of Shape [N,T,H,W,C] (Implementation involves 'Channel Last' Strategy)

        OUTPUTS:-H
        1) X_o : Tensor of shape [N,T,H,W,C]

        """
        X = self.conv2d_depthwise(X)
        X = self.conv2d_pointwise(X)
        X = tf.keras.layers.Reshape((self.H * self.W, self.T, self.filters))(X)
        X = self.conv1d(X)
        X_o = tf.keras.layers.Reshape((self.T, self.H, self.W, self.filters))(X)

        return X_o


class Cross_MSECA_Module(tf.keras.layers.Layer):
    """ Implementation of 3D MSECA(Multi-Scale Efficient Channel Attention) Module """

    def __init__(self, T, H, W, C, k):
        ##### Defining Essentials

        super().__init__()
        self.T = T
        self.H = H
        self.W = W
        self.C = C
        self.k = k  # Kernel dims

        ##### Defining Layers

        #### Adaptive Kernel Size Selection
        # t = int(abs((math.log2(self.C)+self.b)/self.gamma))
        # k = t if t%2 else t+1

        #### Convolution Layers
        self.conv_k1 = tf.keras.layers.Conv1D(filters=1, kernel_size=self.k,
                                              padding='same', activation='linear',
                                              use_bias=False)
        self.conv_k2 = tf.keras.layers.Conv1D(filters=1, kernel_size=(self.k) ** 2,
                                              padding='same', activation='linear',
                                              use_bias=False)
        self.conv_k3 = tf.keras.layers.Conv1D(filters=1, kernel_size=(self.k) ** 3,
                                              padding='same', activation='linear',
                                              use_bias=False)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'T': self.T,
            'H': self.H,
            'W': self.W,
            'C': self.C,
            'k': self.k
        })
        return config

    def call(self, X_in):
        """
        Implemetation of MSECA Module

        INPUTS
        1)X_in : Input of Shape - (,T,H,W,C)

        OUTPUTS
        1)X_mseca : Attentioned Output

        """
        X_in_reshaped = tf.keras.layers.Reshape((-1, self.C))(X_in) # Reshaping the Input to 2D
        X = tf.keras.layers.GlobalAveragePooling3D()(X_in)  # Global Average Pooling
        X = tf.keras.layers.Reshape((self.C, 1))(X)  # Resizing Dimensions for 1D Convolution

        X_k1 = self.conv_k1(X)  # Attention Weight Computation with kernel k1
        X_k2 = self.conv_k2(X)  # Attention Weight Computation with kernel k2
        X_k3 = self.conv_k3(X)  # Attention Weight Computation with kernel k3

        X = tf.keras.layers.Add()([X_k1, X_k2, X_k3])  # Adding Multiscale Information
        X_reshaped = tf.keras.layers.Reshape((1, self.C))(X)  # Reshaping the Multiscale Information

        X_map = tf.linalg.matmul(X, X_reshaped)  # Channel map with one-to-one correspondence
        X_attn_map = tf.keras.layers.Softmax(axis=2)(X_map)  # Softmax activation

        X_mseca = tf.linalg.matmul(X_in_reshaped, X_attn_map)  # Activating input by corresponding attention
        X_mseca = tf.keras.layers.Reshape((self.T, self.H, self.W, self.C))(X_mseca)  # Reshaping final output

        return X_mseca

    ###### Arc Loss


class ArcFace(tf.keras.layers.Layer):

    def __init__(self, n_classes, s, m, regularizer):
        super().__init__()
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer
        })
        return config

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True
                                 )

    def call(self, inputs):
        x, y = inputs
        c = tf.keras.backend.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(
            tf.keras.backend.clip(logits, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


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

flatten_temporal = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, x.shape[2], x.shape[3], x.shape[4])))
restore_shape = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 40, x.shape[1], x.shape[2], x.shape[3])))

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
# tf.keras.utils.plot_model(model)

##### Defining Callbacks
# filepath = "./Models/RDAHGR_RDI_5050_Soli.h5"
# checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max')

###### Training the Model
history = model.fit(
    [X_train_rdi, X_train_rai,y_train_onehot], y_train_onehot,
    epochs=50,
    batch_size=2,
    validation_data=([X_dev_rdi, X_dev_rai,y_dev_onehot], y_dev_onehot),
    validation_batch_size=2
)


##### Saving Training Metrics
np.save('exp_1_mesca_early_history.npy', history.history)

# Save only the architecture
model_json = model.to_json()
with open("exp_1_mesca_early_architecture.json", "w") as json_file:
    json_file.write(model_json)

# Save only the weights
model.save_weights("exp_1_mesca_early_weights.h5")

