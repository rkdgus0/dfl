import copy
from copy import deepcopy
from .BASE_MODEL import BASE


class SUB_SCHEDULER(BASE):
    def __init__(self, model, mec_client_mapping, clients):
        super(MEC, self).__init__()
        self.mec_client_mapping = mec_client_mapping
        self.MEC_models = [model.get_weights() for _ in range(len(mec_client_mapping))]
        self.clients = clients
    
    # MEC trains with local devices.
    # Args: mec_id (int): mec index to train
    def train(self, mec_id, local_epoch):
        MEC_MODEL = copy.deepcopy(self.MEC_models[mec_id]) #.get_weights()
        clients = self.mec_client_mapping[mec_id]
        avg_models = []

        for client_idx in clients:
            self.clients.train(client_idx, MEC_MODEL, local_epochs=local_epoch)
            avg_models.append(copy.deepcopy(self.clients.model.get_weights()))
        self.MEC_models[mec_id] = copy.deepcopy(self.average_model(avg_models)) # .set_weights(self.average_model(avg_models))

        del MEC_MODEL
        avg_models.clear()

        return