import copy

from .BASE_MODEL import BASE
import keras
import random
from sklearn.metrics import f1_score
import numpy as np

class SCHEDULER(BASE):
    def __init__(self, model, clients, NUM_CLIENT, connect_mapping, test_data, delay_method, delay_range=0, avg_method='Equal', delay_epoch=0, n_epochs=1, model_decay='Equal'):
        self.model = model
        self.NUM_CLIENT = NUM_CLIENT
        self.connect_mapping = copy.deepcopy(connect_mapping)
        self.init_connect_mapping = copy.deepcopy(tuple(connect_mapping))
        self.delay_method = delay_method
        self.delay_range = delay_range
        self.clients = clients
        self.test_data = test_data
        self.avg_method = avg_method
        self.n_epochs = n_epochs
        self.model_decay = model_decay
        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy()]
        )
        self.CLIENT_models = [model.get_weights() for _ in range(self.NUM_CLIENT)]

    # ----- Set Connection Map(NxN) ----- #
    def set_connect_mapping(self):
        N = self.NUM_CLIENT
        matrix = np.zeros((N, N), dtype=int)
        np.fill_diagonal(matrix, 1)

        for i in range(N):
            for j in range(i+1, N):
                matrix[i][j] = np.random.choice([0, 1])

        matrix = np.maximum(matrix, matrix.T)
        return matrix

    
    def train(self):
        uploaded_models = []
        connected_client = []
        n_connected_client = 0
        connect_mapping = self.set_connect_mapping()
        
        # Connected Client train
        # Unconnected Client: Local train
        for client_idx in range(self.NUM_CLIENT):
            avg_models = []
            for connect_idx in range(self.NUM_CLIENT):
                if self.connect_mapping[client_idx][connect_idx] == 1:
#TODO: CLIENT_MODEL은 get_weight로 가중치를 줘야함. 인덱스는?
#TODO: CLIENT의 weight을 어떻게 줄 수 있을지 정리
                    CLIENT_MODEL = copy.deepcopy(self.clients.model)
                    self.clients.train(connect_idx, CLIENT_MODEL, local_epoch)
                    avg_models.append(copy.deepcopy(self.clients.model.get_weights()))

                    connected_client.append(connect_idx)
                    n_connected_client[client_idx] += 1
                else:
                    pass
            self.CLIENT_models[client_idx] = copy.deepcopy(self.average_model(avg_models))
            avg_models.clear()

            print(f'Connected Clients with {client_idx}-Client: ', *connected_client)
        
#TODO: 여기 아래로 avg_atio 주고, set_weight를 하는 방법 정리
        avg_ratio = self.calc_avg_ratio(uploaded_models, connected_client)
        if self.model_decay == 'Frac':
            for idx, client_idx in enumerate(connected_client):
                avg_ratio[idx] /= (self.init_connect_mapping[client_idx]+1)
        avg_model = self.average_model(uploaded_models, avg_ratio)

        self.model.set_weights(avg_model)
        for client_idx in connected_client:
            self.clients.MEC_models[client_idx] = copy.deepcopy(avg_model) # .set_weights(avg_model)

        uploaded_models.clear()
        avg_model.clear()

        return

    def test(self, model=None):
        test_x, test_y = self.test_data
        # print(self.model.get_weights())
        if model == None:
            return self.model.evaluate(test_x, test_y)
        else:
            return model.evaluate(test_x, test_y)

            # self.model.compile(
            #     loss=self.loss_fn,
            #     optimizer=self.opt,
            #     metrics=[keras.metrics.CategoricalAccuracy()]
            # )
            # model.compile(
            #     loss=keras.losses.CategoricalCrossentropy(),
            #     metrics=[keras.metrics.CategoricalAccuracy()]
            # )

    def f1_test(self, avg_method):
        test_x, test_y = self.test_data
        pred_y = self.model(test_x)
        pred_y = np.argmax(pred_y, axis=1).astype(int)
        test_y = np.argmax(test_y, axis=1).astype(int)
        if 'macro' in avg_method:
            f1_scores = f1_score(test_y, pred_y, average='macro')
        else:
            f1_scores = f1_score(test_y, pred_y, average='micro')
        return f1_scores

    def calc_avg_ratio(self, models, connected_client):
        ratio = []
        if self.avg_method == 'Acc':
            # tmp_model = copy.deepcopy(self.model)
            for model in models:
                self.model.set_weights(model)
                loss, acc = self.test()
                ratio.append(acc)
        elif 'F1' in self.avg_method:
            for model in models:
                self.model.set_weights(model)
                f1_scores = self.f1_test(self.avg_method)
                ratio.append(f1_scores)
        elif self.avg_method == 'n_data':
            for client_idx in connected_client:
                ratio.append(self.num_mec_data[client_idx])

        elif self.avg_method == 'Equal':
            ratio = [1]*len(models)

        return ratio
    def calc_local_epoch(self, client_idx):
        if self.delay_epoch == 0:
            return self.n_epochs # self.init_connect_mapping[client_idx]
        else:
            return max(1, self.init_connect_mapping[client_idx] * self.delay_epoch)
    def set_lr(self, lr):
        self.clients.clients.set_lr(lr)