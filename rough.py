import tensorflow as tf
from tensorflow.keras import layers

class MVFModuleChannelsLast(tf.keras.layers.Layer):
    def __init__(self, alpha=0.5, kernel_size=3):
        super(MVFModuleChannelsLast, self).__init__()
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.activation = layers.ReLU()

    def build(self, input_shape):
        C = input_shape[-1]  # channels_last: (B, T, H, W, C)
        self.C_mvf = int(C * self.alpha)
        self.C_skip = C - self.C_mvf

        # Define convs for each view (using full convs, no groups)
        self.conv_T = layers.Conv3D(
            filters=self.C_mvf,
            kernel_size=(self.kernel_size, 1, 1),
            padding='same',
            use_bias=False
        )
        self.conv_H = layers.Conv3D(
            filters=self.C_mvf,
            kernel_size=(1, self.kernel_size, 1),
            padding='same',
            use_bias=False
        )
        self.conv_W = layers.Conv3D(
            filters=self.C_mvf,
            kernel_size=(1, 1, self.kernel_size),
            padding='same',
            use_bias=False
        )

    def call(self, x):
        # Split input channels
        x_mvf, x_skip = tf.split(x, [self.C_mvf, self.C_skip], axis=-1)

        # Multi-view convs
        out_T = self.conv_T(x_mvf)
        out_H = self.conv_H(x_mvf)
        out_W = self.conv_W(x_mvf)

        # Add + activate
        out = self.activation(out_T + out_H + out_W)

        # Concat with skip connection
        return tf.concat([out, x_skip], axis=-1)

x = tf.random.normal((2, 40, 32, 32, 64))  # Example: batch=2
mvf = MVFModuleChannelsLast(alpha=0.5)
y = mvf(x)
print(y.shape)  # ➜ (2, 40, 32, 32, 64)
