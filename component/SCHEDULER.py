import copy

from .BASE_MODEL import BASE
import keras
import random
from sklearn.metrics import f1_score
import numpy as np

class SCHEDULER(BASE):
    def __init__(self, model, clients, NUM_CLIENT, test_data, avg_method='Equal', n_epochs=1):
        self.model = model
        self.NUM_CLIENT = NUM_CLIENT
        self.clients = clients
        self.test_data = test_data
        self.avg_method = avg_method
        self.n_epochs = n_epochs
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
        connect_mapping = self.set_connect_mapping()
        #print(f"===== Connecting Map =====")
        #print(connect_mapping)
        
        # Connected Client train
        # Unconnected Client: Local train
        # 1. 각자 업데이트 후 연결된 클라이언트에 파라미터 제공
        model_parameter = [[] for _ in range(self.NUM_CLIENT)]
        for client_idx in range(self.NUM_CLIENT):
            model_parameters = self.CLIENT_models[client_idx]
            self.clients.train(client_idx, model_parameters, self.n_epochs)
            model_parameter[client_idx]=copy.deepcopy(self.clients.model.get_weights())

        print("===== Connected Clients =====")
        for client_idx in range(self.NUM_CLIENT):
            model_weights = []
            for connect_idx in range(self.NUM_CLIENT):
                if connect_mapping[client_idx][connect_idx] == 1:
                    model_weights.append(model_parameter[connect_idx])
                    connected_client.append(connect_idx)
                else:
                    pass
            avg_ratio = self.calc_avg_ratio(model_weights, connected_client)
            self.CLIENT_models[client_idx] = copy.deepcopy(self.average_model(model_weights, avg_ratio))
            print(f'{client_idx}-Client: {connected_client}')
            
            model_weights.clear()
            connected_client.clear()

        # 2. 연결 성공시, 각자 업데이트 후 취합 -> 동시성이 없음.
        '''for client_idx in range(self.NUM_CLIENT):
            model_weights = []
            #if any(self.connect_mapping[client_idx][connect_idx] == 1 for connect_idx in range(self.NUM_CLIENT)):
            for connect_idx in range(self.NUM_CLIENT):
                # client(client_idx)와 연결된 client(connected_idx)는 local update(model.fit).
                # client(client_idx)는 model_weight에 connected client weight을 aggregate & average.
                if connect_mapping[client_idx][connect_idx] == 1:
                    #print(f"[SCHEDULER] {client_idx}-device connect with {connect_idx}-device")
                    model_parameters = self.CLIENT_models[connect_idx]
                    self.clients.train(connect_idx, model_parameters, self.n_epochs)
                    model_weights.append(copy.deepcopy(self.clients.model.get_weights()))

                    connected_client.append(connect_idx)
                else:
                    pass
            avg_ratio = self.calc_avg_ratio(model_weights, connected_client)
            self.CLIENT_models[client_idx] = copy.deepcopy(self.average_model(model_weights, avg_ratio))
            print(f'Connected Clients with {client_idx}-Client: {connected_client} Clients')
            
            model_weights.clear()
            connected_client.clear()'''
        return

    def clients_test(self):
        test_result = [{'loss': [], 'acc': []} for _ in range(self.NUM_CLIENT)]
        test_x, test_y = self.test_data
        for client_idx in range(self.NUM_CLIENT):
            model_parameters = self.CLIENT_models[client_idx]
            model = copy.deepcopy(self.model)
            model.set_weights(model_parameters)
            loss, acc = model.evaluate(test_x, test_y, verbose=2)
            test_result[client_idx]['loss'] = loss
            test_result[client_idx]['acc'] = acc
        return test_result

    def test(self):
        test_x, test_y = self.test_data
        return self.model.evaluate(test_x, test_y)

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
        if self.avg_method.lower() == 'Acc':
            for model in models:
                self.model.set_weights(model)
                loss, acc = self.test()
                ratio.append(acc)
        elif 'f1' in self.avg_method.lower():
            for model in models:
                self.model.set_weights(model)
                f1_scores = self.f1_test(self.avg_method)
                ratio.append(f1_scores)
        elif self.avg_method.lower() == 'n_data':
            for connect_idx in connected_client:
                ratio.append(len(self.clients.datasets[connect_idx]['x']))
        elif self.avg_method.lower() == 'equal':
            ratio = [1]*len(models)

        return ratio
        
    def set_lr(self, lr):
        self.clients.set_lr(lr)