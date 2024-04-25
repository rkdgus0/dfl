import copy

from .BASE_MODEL import BASE
import keras
import random
from sklearn.metrics import f1_score
import numpy as np

class CLIENT(BASE):
    def __init__(self, model, mecs, NUM_MEC, mec_delay, test_data, delay_method, delay_range=0, gmodel_avg_method='Equal', delay_epoch=0, n_epochs=1, model_decay='Equal', num_mec_datas=None):
        self.model = model
        self.NUM_MEC = NUM_MEC
        self.mec_delay = copy.deepcopy(mec_delay)
        self.init_mec_delay = copy.deepcopy(tuple(mec_delay))
        self.delay_method = delay_method
        self.delay_range = delay_range
        self.mecs = mecs
        self.test_data = test_data
        self.gmodel_avg_method = gmodel_avg_method
        self.delay_epoch = delay_epoch
        self.n_epochs = n_epochs
        self.model_decay = model_decay
        # for mec_idx in range(self.NUM_MEC):
        #     self.mecs.MEC_models[mec_idx] = self.model.get_weights() # = .set_weights(self.model.get_weights())
        self.num_mec_data = copy.deepcopy(num_mec_datas)
        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy()]
        )

    def train(self):
        uploaded_models = []
        participating_mec = []
        for mec_idx in range(self.NUM_MEC):
            if self.mec_delay[mec_idx] == 0:
                local_epoch = self.calc_local_epoch(mec_idx)
                self.mecs.train(mec_idx, local_epoch)

                participating_mec.append(mec_idx)
                uploaded_models.append(copy.deepcopy(self.mecs.MEC_models[mec_idx])) # .get_weights())
                self.mec_delay[mec_idx] = self.set_mec_delay(mec_idx)
            else:
                self.mec_delay[mec_idx] -= 1

        print('Participating MEC: ', *participating_mec)
        if len(participating_mec) == 0:
            return

        avg_ratio = self.calc_avg_ratio(uploaded_models, participating_mec)
        if self.model_decay == 'Frac':
            for idx, mec_idx in enumerate(participating_mec):
                avg_ratio[idx] /= (self.init_mec_delay[mec_idx]+1)
        avg_model = self.average_model(uploaded_models, avg_ratio)

        self.model.set_weights(avg_model)
        for mec_idx in participating_mec:
            self.mecs.MEC_models[mec_idx] = copy.deepcopy(avg_model) # .set_weights(avg_model)

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

    def f1_test(self, gmodel_avg_method):
        test_x, test_y = self.test_data
        pred_y = self.model(test_x)
        pred_y = np.argmax(pred_y, axis=1).astype(int)
        test_y = np.argmax(test_y, axis=1).astype(int)
        if 'macro' in gmodel_avg_method:
            f1_scores = f1_score(test_y, pred_y, average='macro')
        else:
            f1_scores = f1_score(test_y, pred_y, average='micro')
        return f1_scores

    def set_mec_delay(self, mec_idx):
        if self.delay_method == 'Fixed':
            return self.init_mec_delay[mec_idx]# + 1
        elif self.delay_method == 'Range':
            dr = self.delay_range
            return max(0, self.init_mec_delay[mec_idx] + random.randint(-dr, dr))# + 1

    def calc_avg_ratio(self, models, participating_mec):
        ratio = []
        if self.gmodel_avg_method == 'Acc':
            # tmp_model = copy.deepcopy(self.model)
            for model in models:
                self.model.set_weights(model)
                loss, acc = self.test()
                ratio.append(acc)
        elif 'F1' in self.gmodel_avg_method:
            for model in models:
                self.model.set_weights(model)
                f1_scores = self.f1_test(self.gmodel_avg_method)
                ratio.append(f1_scores)
        elif self.gmodel_avg_method == 'n_data':
            for mec_idx in participating_mec:
                ratio.append(self.num_mec_data[mec_idx])

        elif self.gmodel_avg_method == 'Equal':
            ratio = [1]*len(models)

        elif self.gmodel_avg_method == 'FedAT':
            at_ratio = [1 + x for x in list(self.init_mec_delay)]
            at_ratio *= self.delay_epoch + 1
            for mec_idx in participating_mec:
                ratio.append(at_ratio[mec_idx])
        return ratio
    def calc_local_epoch(self, mec_idx):
        if self.delay_epoch == 0:
            return self.n_epochs # self.init_mec_delay[mec_idx]
        else:
            return max(1, self.init_mec_delay[mec_idx] * self.delay_epoch)
    def set_lr(self, lr):
        self.mecs.clients.set_lr(lr)