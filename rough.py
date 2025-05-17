import tensorflow as tf
from tensorflow.keras import layers, models


class MS_CAM_3D(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=16):
        super(MS_CAM_3D, self).__init__()
        self.channels = channels
        self.reduction = reduction

        # Global context layers
        self.global_fc1 = layers.Dense(channels // reduction, activation='relu')
        self.global_bn1 = layers.BatchNormalization()
        self.global_fc2 = layers.Dense(channels)
        self.global_bn2 = layers.BatchNormalization()

        # Local context layers (3D convs)
        self.local_conv1 = layers.Conv3D(channels // reduction, 1, padding='same')
        self.local_bn1 = layers.BatchNormalization()
        self.local_conv2 = layers.Conv3D(channels, 1, padding='same')
        self.local_bn2 = layers.BatchNormalization()

        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        # Global context
        gap = tf.reduce_mean(x, axis=[1, 2, 3])  # Global Avg Pooling over D, H, W
        g = self.global_fc1(gap)
        g = self.global_bn1(g)
        g = self.global_fc2(g)
        g = self.global_bn2(g)
        g = tf.reshape(g, [-1, 1, 1, 1, self.channels])  # for broadcasting

        # Local context
        l = self.local_conv1(x)
        l = self.local_bn1(l)
        l = tf.nn.relu(l)
        l = self.local_conv2(l)
        l = self.local_bn2(l)

        # Combine global and local
        combined = self.sigmoid(g + l)
        return x * combined


class iAFF_3D(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=16):
        super(iAFF_3D, self).__init__()
        self.ms_cam1 = MS_CAM_3D(channels, reduction)
        self.ms_cam2 = MS_CAM_3D(channels, reduction)

    def call(self, x, y):
        # Step 1: Initial fusion and weighting
        fuse1 = x + y
        w1 = self.ms_cam1(fuse1)
        z1 = x * w1 + y * (1.0 - w1)

        # Step 2: Refined fusion and weighting
        w2 = self.ms_cam2(z1)
        z2 = x * w2 + y * (1.0 - w2)

        return z2


# Inputs
rai = tf.keras.Input(shape=(40, 32, 32, 64))  # 5D input
rdi = tf.keras.Input(shape=(40, 32, 32, 64))

# Fuse them with iAFF_3D
iaff = iAFF_3D(channels=64, reduction=16)
fused = iaff(rai, rdi)

# Optional: Wrap in a model
model = tf.keras.Model(inputs=[rai, rdi], outputs=fused)

# Generate dummy data (e.g., batch size 2)
batch_size = None
rai_sample = tf.random.normal((batch_size, 40, 32, 32, 64))
rdi_sample = tf.random.normal((batch_size, 40, 32, 32, 64))

# Pass the samples through the model
fused_result = model([rai_sample, rdi_sample])

# Show shape
print("Output shape:", fused_result.shape)