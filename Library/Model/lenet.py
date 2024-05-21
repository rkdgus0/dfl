import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

class McMahanTwoNN(Model):
    def __init__(self, input_shape):
        super(McMahanTwoNN, self).__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=42)
        self.flatten = Flatten(input_shape=(32, 32, 3))
        self.dense1 = Dense(64, activation='relu', kernel_initializer=initializer)
        self.dense2 = Dense(10)
        self.dense3 = Dense(10, activation='softmax')

        self.build((None,) + tuple(input_shape))

    def call(self, x, training=None, mask=None) -> Model:
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return self.dense3(x)

class McMahanCNN(Model):
    def __init__(self, input_shape):
        super(McMahanCNN, self).__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=42)
        self.conv1 = Conv2D(32, 5, activation='relu', padding='same', kernel_initializer=initializer)
        self.pool1 = MaxPool2D(2, strides=2)
        self.conv2 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer=initializer)
        self.pool2 = MaxPool2D(2, strides=2)
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu', kernel_initializer=initializer)
        self.dense2 = Dense(10)
        self.dense3 = Dense(10, activation='softmax')

        self.build((None,) + input_shape)

    def call(self, x, training=None, mask=None) -> Model:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return self.dense3(x)
    
    @classmethod
    def from_config(cls, config):
        input_shape = tuple(config['input_shape'])
        return cls(input_shape=input_shape)