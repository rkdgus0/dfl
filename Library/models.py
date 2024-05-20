import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras import Model

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers import *

def resnet_block(inputs, num_filters, kernel_size=3, stride=1, conv_first=True):
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=stride, padding='same',
                  kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    else:
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('Depth should be 6n+2 (e.g. 20, 32, 44).')
    num_filters = 64
    num_res_blocks = int((depth - 2) / 6)

    inputs = layers.Input(shape=input_shape)
    x = resnet_block(inputs, num_filters)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # First layer but not first stack
                strides = 2  # downsample
            y = resnet_block(x, num_filters, stride=strides)
            y = resnet_block(y, num_filters, conv_first=False)
            if stack > 0 and res_block == 0:  # First layer but not first stack
                x = Conv2D(num_filters, kernel_size=1, strides=strides, padding='same',
                           kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            x = layers.add([x, y])
        num_filters *= 2

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


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

def define_model(args):
    model_name = args.model.lower()

    if args.pre_trained:
        weight = 'imagenet'
    else:
        weight = None

    num_classes = 10
    input_shape = (32, 32, 3)

    Model = Sequential()
    Model.add(InputLayer(input_shape=(32,32,3)))
    Model.add(Normalization(mean=[0.4914, 0.4822, 0.4465], variance=[0.2023, 0.1994, 0.2010]))
    
    if model_name == "mcmahan2nn":
        Model = McMahanTwoNN(input_shape=input_shape)

    elif model_name =="mcmahancnn":
        Model = McMahanCNN(input_shape=input_shape)

    elif model_name == "resnet18":
        Model = resnet_v1(input_shape=input_shape, depth=20)
        
    elif model_name in ['resnet50', 'resnet101', 'densenet121']:
        if model_name == 'resnet50':
            model = ResNet50(weights=weight, include_top=False, input_shape=input_shape)
        elif model_name == 'resnet101':
            model = ResNet101(weights=weight, include_top=False, input_shape=input_shape)
        elif model_name == 'densenet121':
            model = DenseNet121(weights=weight, include_top=False, input_shape=input_shape)
        Model.add(model)
        Model.add(GlobalAveragePooling2D())
        Model.add(Dense(num_classes, activation='softmax'))
    
    else:
        raise ValueError("check model name")
    
    
    return Model