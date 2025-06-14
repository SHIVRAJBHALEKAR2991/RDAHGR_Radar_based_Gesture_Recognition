import tensorflow as tf

class SE_Block(tf.keras.layers.Layer):
    """
    Squeeze-and-Excitation (SE) Block — from SE-LPN-DPFF paper
    Replaces your Cross_MSECA_Module (Multi-Scale Channel Attention)

    Input:
        Tensor of shape (None, T=40, H=32, W=32, C=128)
    Output:
        Tensor of same shape with recalibrated channels
    """

    def __init__(self, reduction_ratio=2, name='se_block'):
        super(SE_Block, self).__init__(name=name)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        C = input_shape[-1]
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling3D()
        self.fc1 = tf.keras.layers.Dense(C // self.reduction_ratio,
                                         activation='relu',
                                         kernel_initializer='he_normal',
                                         use_bias=True)
        self.fc2 = tf.keras.layers.Dense(C,
                                         activation='sigmoid',
                                         kernel_initializer='he_normal',
                                         use_bias=True)
        self.reshape = tf.keras.layers.Reshape((1, 1, 1, C))

    def call(self, inputs):
        # Squeeze: global context
        x = self.global_avg_pool(inputs)  # Shape: (None, C)
        x = self.fc1(x)
        x = self.fc2(x)  # Channel attention vector
        x = self.reshape(x)  # Reshape to (None, 1, 1, 1, C)

        # Excitation: scale original features
        scaled = inputs * x  # Channel-wise multiplication (broadcasted)
        return scaled
