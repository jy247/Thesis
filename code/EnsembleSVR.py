import pandas as pd
import numpy as np
import copy
from numpy.random import randint

class EnsembleSVR():

    def __init__(self, the_model, n_total=17, n_each=10, n_ensemble=20):
        #generate random indices
        self.ensemble_indices = randint(0,n_total-1, [n_ensemble,n_each])
        self.models = []
        for i in range(n_ensemble):
            self.models.append(copy.deepcopy(the_model))

    def predict(self, xtest):
        n_test = len(xtest)
        results = pd.DataFrame(index=range(n_test))

        for i in range(self.ensemble_indices.shape[0]):
            one_results = self.models[i].predict(xtest[:,self.ensemble_indices[0]])
            #one_results = np.reshape(one_results,[1, n_test])
            results[i] = one_results

        return np.mean(results,axis=1)

    def fit(self, xtrain, ytrain):
        for i in range(self.ensemble_indices.shape[0]):
            self.models[i].fit(xtrain[:,self.ensemble_indices[0]], ytrain)

    def copy(self):
        new_copy = EnsembleSVR(self.models[0])
        return new_copy

    __copy__ = copy  # Now works with copy.copy too


