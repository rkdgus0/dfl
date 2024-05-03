from copy import deepcopy
from .BASE_MODEL import BASE
import keras
import numpy as np

class USER(BASE):
    def __init__(self, args, model, datasets, epochs, batch_size, device):
        self.datasets = datasets
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = deepcopy(model)
        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=keras.optimizers.SGD(learning_rate=args.lr, clipvalue=1.0)
        )

    def train(self, client_idx, model_parameters, local_epochs=1):
        """Single round of local training for one client.

        Note:
            Load dataset using client_idx and train model with model_parameters.

        Args:
            client_idx (int): client index to train
            model_parameters (tf.Tensor): serialized model parameters.
        """
        self.model.set_weights(model_parameters)

        self.model.fit(self.datasets[client_idx]['x'], self.datasets[client_idx]['y'],
                       epochs=local_epochs, batch_size=self.batch_size, verbose=1)
        return

        # self.model.compile(
        #     loss=self.criterion,
        #     optimizer=self.optimizer,
        # )
        # if self.datasets[client_idx]['x'][0].shape[-1] == 1:
        #     # x_data = np.repeat(self.datasets[client_idx]['x'], 3, axis=-1)
        #     y_data = np.array(self.datasets[client_idx]['y'])
        #     self.model.fit(x_data, y_data, epochs=local_epochs, batch_size=32)
        # else:
    def set_lr(self, lr):
        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=keras.optimizers.SGD(learning_rate=lr, clipvalue=1.0)
        )