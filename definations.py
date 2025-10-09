import tensorflow as tf
import math


####### TEA Module #######

class TEA_ME(tf.keras.layers.Layer):
    """ TEA Module's Motion Excitation Block for Motion Modelling """

    def __init__(self, reduction_factor, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.reduction_factor = reduction_factor
        self.num_channels = num_channels

        red_val = int(self.num_channels // self.reduction_factor)
        self.conv_red = tf.keras.layers.Conv2D(
            filters=red_val, kernel_size=(1, 1), padding='same',
            activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )

        self.conv_transform = tf.keras.layers.Conv2D(
            filters=red_val, kernel_size=(3, 3), padding='same',
            groups=red_val, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )

        self.conv_exp = tf.keras.layers.Conv2D(
            filters=self.num_channels, kernel_size=(1, 1), padding='same',
            activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction_factor': self.reduction_factor,
            'num_channels': self.num_channels,
        })
        return config

    def call(self, X):
        N, T, H, W, C = X.shape
        X_red = self.conv_red(X)
        X_red_M1 = X_red[:, :-1, :, :, :]
        X_red_M2 = X_red[:, 1:, :, :, :]
        X_transform = self.conv_transform(X_red_M2)
        M = tf.keras.layers.Add()([X_transform, -X_red_M1])
        M = tf.keras.layers.ZeroPadding3D(((1, 0), (0, 0), (0, 0)))(M)

        Ms = tf.keras.layers.AveragePooling3D(pool_size=(1, H, W))(M)
        Ms_expanded = self.conv_exp(Ms)

        A = 2 * (tf.keras.activations.sigmoid(Ms_expanded)) - 1
        X_bar = tf.math.multiply(X, A)
        X_o = tf.keras.layers.Add()([X, X_bar])
        return X_o


class TEA_MTA(tf.keras.layers.Layer):
    def __init__(self, N, T, H, W, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.N = N
        self.T = T
        self.H = H
        self.W = W
        self.split_factor = self.num_channels // 4

        self.temp_conv1_layers = [
            tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding='same', activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-5))
            for _ in range(self.split_factor)
        ]
        self.temp_conv2_layers = [
            tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding='same', activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-5))
            for _ in range(self.split_factor)
        ]
        self.temp_conv3_layers = [
            tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding='same', activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-5))
            for _ in range(self.split_factor)
        ]

        self.conv_spa_1 = tf.keras.layers.Conv2D(filters=self.split_factor, kernel_size=(3, 3), padding='same',
                                                 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.conv_spa_2 = tf.keras.layers.Conv2D(filters=self.split_factor, kernel_size=(3, 3), padding='same',
                                                 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.conv_spa_3 = tf.keras.layers.Conv2D(filters=self.split_factor, kernel_size=(3, 3), padding='same',
                                                 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_channels': self.num_channels,
            'N': self.N,
            'T': self.T,
            'H': self.H,
            'W': self.W
        })
        return config

    def grouped_conv1d(self, x, conv_layers):
        x_splits = tf.split(x, num_or_size_splits=self.split_factor, axis=-1)
        out_splits = [conv_layer(split) for conv_layer, split in zip(conv_layers, x_splits)]
        return tf.concat(out_splits, axis=-1)

    def call(self, X):
        batch_size = tf.shape(X)[0]
        T, H, W, split_factor = self.T, self.H, self.W, self.split_factor

        Xi_0, Xi_1, Xi_2, Xi_3 = tf.split(X, num_or_size_splits=4, axis=-1)

        Xo_0 = Xi_0

        Xi_1 = tf.keras.layers.Add()([Xo_0, Xi_1])
        Xi_1_temp = self.grouped_conv1d(tf.reshape(Xi_1, [batch_size * T, H * W, split_factor]),
                                        self.temp_conv1_layers)
        Xo_1 = self.conv_spa_1(tf.reshape(Xi_1_temp, [batch_size * T, H, W, split_factor]))
        Xo_1 = tf.reshape(Xo_1, [batch_size, T, H, W, split_factor])

        Xi_2 = tf.keras.layers.Add()([Xo_1, Xi_2])
        Xi_2_temp = self.grouped_conv1d(tf.reshape(Xi_2, [batch_size * T, H * W, split_factor]),
                                        self.temp_conv2_layers)
        Xo_2 = self.conv_spa_2(tf.reshape(Xi_2_temp, [batch_size * T, H, W, split_factor]))
        Xo_2 = tf.reshape(Xo_2, [batch_size, T, H, W, split_factor])

        Xi_3 = tf.keras.layers.Add()([Xo_2, Xi_3])
        Xi_3_temp = self.grouped_conv1d(tf.reshape(Xi_3, [batch_size * T, H * W, split_factor]),
                                        self.temp_conv3_layers)
        Xo_3 = self.conv_spa_3(tf.reshape(Xi_3_temp, [batch_size * T, H, W, split_factor]))
        Xo_3 = tf.reshape(Xo_3, [batch_size, T, H, W, split_factor])

        return tf.keras.layers.Concatenate(axis=-1)([Xo_0, Xo_1, Xo_2, Xo_3])


class CT_Module(tf.keras.layers.Layer):
    """ 3D Tensor Separable Convolution """

    def __init__(self, T, H, W, C, **kwargs):
        super().__init__(**kwargs)
        self.T, self.H, self.W, self.C = T, H, W, C

        K = int(math.log2(C))
        k1_dim = int(K / 2)
        self.k1 = int(2 ** k1_dim)
        self.k2 = int(2 ** (K - k1_dim))

        self.conv_k1 = tf.keras.layers.Conv3D(filters=self.C, kernel_size=(3, 3, 3), padding='same',
                                              activation='linear', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.conv_k2 = tf.keras.layers.Conv3D(filters=self.C, kernel_size=(3, 3, 3), padding='same',
                                              activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))

    def get_config(self):
        config = super().get_config()
        config.update({'T': self.T, 'H': self.H, 'W': self.W, 'C': self.C})
        return config

    def call(self, X0):
        if X0.shape[-1] != self.C:
            raise ValueError(f"Input channels ({X0.shape[-1]}) do not match expected channels ({self.C}).")

        X0 = tf.keras.layers.Reshape((self.T, self.H, self.W, self.k1 * self.k2))(X0)
        X1 = self.conv_k1(X0)
        X2 = self.conv_k2(X1)
        return tf.keras.layers.Reshape((self.T, self.H, self.W, self.C))(X2)


class two_plus_oneDConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_dims, H, W, C, T, **kwargs):
        super().__init__(**kwargs)
        self.filters, self.kernel_dims, self.H, self.W, self.C, self.T = filters, kernel_dims, H, W, C, T

        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(self.kernel_dims, self.kernel_dims), padding='same',
            activation='linear', depth_multiplier=1
        )
        self.pointwise_conv = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1),
                                                     padding='same', activation='relu',
                                                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.conv1d = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_dims,
                                             padding='same', activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(1e-5))

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'kernel_dims': self.kernel_dims,
                       'H': self.H, 'W': self.W, 'C': self.C, 'T': self.T})
        return config

    def call(self, X):
        X_reshaped = tf.reshape(X, [-1, self.H, self.W, self.C])
        X_conv2d = self.depthwise_conv(X_reshaped)
        X_conv2d = self.pointwise_conv(X_conv2d)
        X_conv2d = tf.reshape(X_conv2d, [-1, self.T, self.H * self.W, self.filters])
        X_conv2d = tf.transpose(X_conv2d, perm=[0, 2, 1, 3])
        X_flat = tf.reshape(X_conv2d, [-1, self.T, self.filters])
        X_conv1d = self.conv1d(X_flat)
        X_conv1d = tf.reshape(X_conv1d, [-1, self.H * self.W, self.T, self.filters])
        X_conv1d = tf.transpose(X_conv1d, perm=[0, 2, 1, 3])
        return tf.reshape(X_conv1d, [-1, self.T, self.H, self.W, self.filters])


class Cross_MSECA_Module(tf.keras.layers.Layer):
    """ Implementation of 3D MSECA """

    def __init__(self, T, H, W, C, k, **kwargs):
        super().__init__(**kwargs)
        self.T, self.H, self.W, self.C, self.k = T, H, W, C, k

        self.conv_k1 = tf.keras.layers.Conv1D(filters=1, kernel_size=self.k,
                                              padding='same', activation='linear', use_bias=False)
        self.conv_k2 = tf.keras.layers.Conv1D(filters=1, kernel_size=self.k ** 2,
                                              padding='same', activation='linear', use_bias=False)
        self.conv_k3 = tf.keras.layers.Conv1D(filters=1, kernel_size=self.k ** 3,
                                              padding='same', activation='linear', use_bias=False)

    def get_config(self):
        config = super().get_config()
        config.update({'T': self.T, 'H': self.H, 'W': self.W, 'C': self.C, 'k': self.k})
        return config

    def call(self, X_in):
        X_in_reshaped = tf.keras.layers.Reshape((-1, self.C))(X_in)
        X = tf.keras.layers.GlobalAveragePooling3D()(X_in)
        X = tf.keras.layers.Reshape((self.C, 1))(X)
        X = tf.keras.layers.Add()([self.conv_k1(X), self.conv_k2(X), self.conv_k3(X)])
        X_map = tf.linalg.matmul(X, tf.keras.layers.Reshape((1, self.C))(X))
        X_attn_map = tf.keras.layers.Softmax(axis=2)(X_map)
        X_mseca = tf.linalg.matmul(X_in_reshaped, X_attn_map)
        return tf.keras.layers.Reshape((self.T, self.H, self.W, self.C))(X_mseca)


class ArcFace(tf.keras.layers.Layer):
    def __init__(self, n_classes, s, m, regularizer, **kwargs):
        super().__init__(**kwargs)
        self.n_classes, self.s, self.m = n_classes, s, m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def get_config(self):
        config = super().get_config()
        config.update({'n_classes': self.n_classes, 's': self.s,
                       'm': self.m, 'regularizer': self.regularizer})
        return config

    def build(self, input_shape):
        super().build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs):
        x, y = inputs
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = tf.matmul(x, W)
        theta = tf.acos(tf.keras.backend.clip(logits, -1.0 + tf.keras.backend.epsilon(),
                                              1.0 - tf.keras.backend.epsilon()))
        target_logits = tf.cos(theta + self.m)
        logits = logits * (1 - y) + target_logits * y
        logits *= self.s
        return tf.nn.softmax(logits)

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)
