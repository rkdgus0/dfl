import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras import Model

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers import *

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

'''def define_model(args):
    num_classes = 10
    input_shape = (32, 32, 3)
    model_name = args.model.lower()
    dataset = args.dataset

    if args.pre_trained:
        weight = 'imagenet'
    else:
        weight = None
    
    if model_name == 'resnet50':
        model = ResNet50(weights=weight, include_top=False, input_shape=input_shape)
    elif model_name == 'resnet101':
        model = ResNet101(weights=weight, include_top=False, input_shape=input_shape)
    elif model_name == 'densenet121':
        model = DenseNet121(weights=weight, include_top=False, input_shape=input_shape)
    else:
        raise ValueError("check model name")

    Model = Sequential()
    Model.add(InputLayer(input_shape=(32,32,3)))
    Model.add(Normalization(mean=[0.4914, 0.4822, 0.4465], variance=[0.2023, 0.1994, 0.2010]))
    Model.add(model)
    Model.add(GlobalAveragePooling2D())
    Model.add(Dense(num_classes, activation='softmax'))

    return Model
'''
def define_model(args):
    model_name = args.model.lower()

    if args.pre_trained:
        weight = 'imagenet'
    else:
        weight = None

    num_classes = 10
    input_shape = (32, 32, 3)


    if model_name == "mcmahan2nn":
        Model = McMahanTwoNN(input_shape=input_shape)

    elif model_name =="mcmahancnn":
        Model = McMahanCNN(input_shape=input_shape)

    elif model_name in ['resnet50', 'resnet101', 'densenet121']:
        Model = Sequential()
        Model.add(InputLayer(input_shape=(32,32,3)))
        Model.add(Normalization(mean=[0.4914, 0.4822, 0.4465], variance=[0.2023, 0.1994, 0.2010]))
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