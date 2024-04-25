import numpy as np

class BASE:
    def __int__(self):
        pass

    @staticmethod
    def average_model(models, avg_ratio=None):
        new_weights = list()

        if avg_ratio == None:
            for weights_list_tuple in zip(*models):
                new_weights.append(
                    np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
                )
        elif avg_ratio:
            for weights_list_tuple in zip(*models):
                new_weights.append(
                    np.array([np.average(np.array(w), weights=avg_ratio, axis=0) for w in zip(*weights_list_tuple)])
                )
        return new_weights