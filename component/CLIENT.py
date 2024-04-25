from copy import deepcopy
from .BASE_MODEL import BASE
import keras
import numpy as np

#TODO: 오류 잡으면서 해보기. weight 제대로 들어가는지 확인
class CLIENT(BASE):
    def __init__(self, args, NUM_CLIENT, model, datasets, epochs, batch_size):
        self.NUM_CLIENT
        self.datasets = datasets
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = deepcopy(model)
        self.set_model(args.lr, args.opt)
        self.CLIENT_models = [model.get_weights() for _ in range(self.NUM_CLIENT)]

    # ----- Local train ----- #
    # Load dataset using client_idx/train model(with model_parameters)
    # client_idx(int): client index
    # model_parameters(tf.Tensor): serialized model parameters
    def train(self, client_idx, model_parameters, local_epochs=1):
        self.model.set_weights(model_parameters)

        self.model.fit(self.datasets[client_idx]['x'], self.datasets[client_idx]['y'],
                       epochs=local_epochs, batch_size=self.batch_size, verbose=1)
        return

    def set_model(self, lr, opt=args.opt):
        if opt.lower() == 'sgd':
            optim = keras.optimizers.SGD(learning_rate=lr, clipvalue=1.0)
        elif opt.lower() == 'adam':
            optim = keras.optimizers.Adam(learning_rate=lr, clipvalue=1.0)
        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=optim
        )
        return