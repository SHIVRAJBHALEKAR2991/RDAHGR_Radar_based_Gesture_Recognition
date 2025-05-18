import tensorflow as tf

# Define dummy input: [batch, T, H, W, C]
N, T, H, W, C = 2, 40, 32, 32, 1
dummy_input = tf.random.normal([N, T, H, W, C])

# Define the custom layer
class two_plus_oneDConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_dims, H, W, C, T, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_dims = kernel_dims
        self.H = H
        self.W = W
        self.C = C
        self.T = T

        self.conv2d_depthwise = tf.keras.layers.Conv2D(
            filters=C,
            kernel_size=(self.kernel_dims, self.kernel_dims),
            padding='same',
            activation='linear',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )

        self.conv2d_pointwise = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )

        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_dims,
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )

    def call(self, X):
        N = tf.shape(X)[0]
        X = tf.reshape(X, [-1, self.H, self.W, self.C])
        X = self.conv2d_depthwise(X)
        X = self.conv2d_pointwise(X)
        X = tf.reshape(X, [N, self.T, self.H, self.W, self.filters])
        X = tf.transpose(X, [0, 2, 3, 1, 4])
        X = tf.reshape(X, [-1, self.T, self.filters])
        X = self.conv1d(X)
        X = tf.reshape(X, [N, self.H, self.W, self.T, self.filters])
        X = tf.transpose(X, [0, 3, 1, 2, 4])
        return X

# Instantiate and run
conv_layer = two_plus_oneDConv(32, 3, H, W, C, T)
output = conv_layer(dummy_input)
print("Output shape:", output.shape)
