import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import ResNet101

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers import *


def define_model(args):
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

