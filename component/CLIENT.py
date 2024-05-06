from copy import deepcopy
from .BASE_MODEL import BASE
import keras
import numpy as np
#import tensorflow as tf
#from keras.losses import CategoricalCrossentropy
#from tensorflow.python.keras.optimizers import gradient_descent_v2, adam_v2

#TODO: 오류 잡으면서 해보기. weight 제대로 들어가는지 확인
class CLIENT(BASE):
    def __init__(self, opt, lr, model, datasets, epochs, batch_size, device):
        self.datasets = datasets
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = deepcopy(model)
        self.origin_lr = lr
        self.opt = opt.lower()
        if self.opt == 'sgd':
            self.model.compile(
                loss=keras.losses.CategoricalCrossentropy(),
                optimizer=keras.optimizers.SGD(learning_rate=self.origin_lr, clipvalue=1.0)
            )
        elif self.opt == 'adam':
            self.model.compile(
                loss=keras.losses.CategoricalCrossentropy(),
                optimizer=keras.optimizers.Adam(learning_rate=self.origin_lr, clipvalue=1.0)
            )
        #self.model.compile(
        #    loss=CategoricalCrossentropy(),
        #    optimizer=gradient_descent_v2.SGD(learning_rate=self.origin_lr, clipvalue=1.0)
        #)

    # ----- Local train ----- #
    # Load dataset using client_idx/train model(with model_parameters)
    # client_idx(int): client index
    # model_parameters(tf.Tensor): serialized model parameters
    def train(self, client_idx, model_parameters, local_epochs=1):
        self.model.set_weights(model_parameters)
        self.model.fit(self.datasets[client_idx]['x'], self.datasets[client_idx]['y'],
                       epochs=local_epochs, batch_size=self.batch_size, verbose=1)
        return

    def set_lr(self, lr):
        '''self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=keras.optimizers.SGD(learning_rate=lr, clipvalue=1.0))
        '''
        if self.opt == 'sgd':
            self.model.compile(
                loss=keras.losses.CategoricalCrossentropy(),
                optimizer=keras.optimizers.SGD(learning_rate=lr, clipvalue=1.0)
            )
        elif self.opt == 'adam':
            self.model.compile(
                loss=keras.losses.CategoricalCrossentropy(),
                optimizer=keras.optimizers.Adam(learning_rate=lr, clipvalue=1.0)
            )