import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

import ModelHelper as mh
np.random.seed(123456789)
NUM_WEIGHT_BRACKETS = 4

class EnsembleSVR():

    def __init__(self, model_type, n_total=17, n_each=10, n_ensemble=50):

        if model_type != mh.LIN:
            raise ValueError('Ensemble SVR only handles Linear kernel!')
        self.model_weights = np.ones([n_ensemble]) / n_ensemble
        self.models = []
        self.n_ensemble = n_ensemble

        #generate random indices
        self.ensemble_indices = []
        for i in range(n_ensemble):
            self.ensemble_indices.append(np.random.choice(n_total, size=n_each, replace=False))

        random_c_exp = np.random.rand(n_ensemble) * 3
        random_epsilon = np.random.rand(n_ensemble) * 0.4
        for i in range(n_ensemble):
            c = 0.1 * 10 ** random_c_exp[i]
            epsilon = random_epsilon[i]
            model = mh.get_model(mh.LIN, c, epsilon)
            #model = mh.get_model_fixed(mh.LIN)
            self.models.append(model)


    #set the model weights based on the scores
    def reweight(self, x_train, y_train):
        # weight by cross validation scores
        all_scores = []
        self.model_weights = np.zeros([self.n_ensemble])
        for i in range(self.n_ensemble):
            model = self.models[i]
            pipeline = make_pipeline(preprocessing.StandardScaler(), model)
            score = sum(cross_val_score(pipeline, x_train.iloc[:,self.ensemble_indices[i]], y_train, cv=5, scoring='r2'))
            all_scores.append(score)

        sorted_indices = np.argsort(all_scores)
        weight_index = 1
        num_per_bracket = self.n_ensemble / NUM_WEIGHT_BRACKETS
        for i in range(self.n_ensemble):
            if weight_index * num_per_bracket < i:
                weight_index += 1

            self.model_weights[sorted_indices[i]] = weight_index - 1

        #make them sum to 1
        total_weight = np.sum(self.model_weights)
        self.model_weights = self.model_weights / total_weight


    def predict(self, xtest):
        n_test = len(xtest)
        results = pd.DataFrame(index=range(n_test))

        for i in range(self.n_ensemble):
            one_results = self.models[i].predict(xtest[:,self.ensemble_indices[i]])
            #one_results = self.models[i].predict(xtest)
            #one_results = np.reshape(one_results,[1, n_test])
            results[i] = one_results
            #print(one_results)

        return np.dot(results,self.model_weights)

    def fit(self, xtrain, ytrain):

        for i in range(self.n_ensemble):
            data_indices = np.random.choice(xtrain.shape[0], size=60, replace=False)
            only_data = xtrain[data_indices,:]
            #self.models[i].fit(only_data[:,self.ensemble_indices[i]], ytrain[data_indices])
            self.models[i].fit(xtrain[:, self.ensemble_indices[i]], ytrain)
            #self.models[i].fit(xtrain, ytrain)

    # def copy(self):
    #     new_copy = EnsembleSVR(self.models[0])
    #     return new_copy
    #
    # __copy__ = copy  # Now works with copy.copy too
    #
    #
