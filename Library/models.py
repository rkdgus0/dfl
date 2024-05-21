import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import ResNet101

from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, InputLayer, Normalization

from Library.Model.resnet import resnet18
from Library.Model.lenet import McMahanTwoNN, McMahanCNN

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
        Model = resnet18(input_shape=input_shape, depth=20)
        
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