import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
from sklearn import kernel_ridge
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import kernels
from timeit import default_timer as timer
from sklearn.tree import DecisionTreeRegressor

import code.DataHelper as dh
import datetime
import dateutil

LIN = 0
RBF = 1
POLY2 = 2
POLY3 = 3
POLY4 = 4
Rand_F = 5
GPR = 6
SGD = 7
KRR = 8
DT = 9
gp_kernel = kernels.DotProduct() \
            + kernels.WhiteKernel(1e-1)

MODELS_DIC ={
    #RBF:svm.SVR(kernel='rbf', C=8, epsilon=0.1, gamma=0.001),
    #RBF: svm.SVR(kernel='rbf', C=12, epsilon=0.16, gamma=0.0001),
    RBF: svm.SVR(kernel='rbf', C=50, epsilon=0.18, gamma=0.00012),
    POLY2:svm.SVR(kernel='poly', C=0.2, degree=2, epsilon=0.25),
    POLY3:svm.SVR(kernel='poly', C=0.12, degree=3, epsilon=0.33),
    POLY4:svm.SVR(kernel='poly', C=0.07, degree=4, epsilon=1.2),
    #LIN:svm.SVR(kernel='linear', C=5, epsilon=0.3),
    #LIN:svm.SVR(kernel='linear', C=120, epsilon=0.25),
    LIN:svm.SVR(kernel='linear', C=7, epsilon=0.2), #ensemble
    DT:DecisionTreeRegressor(),
    #LIN:svm.LinearSVR(C=1, epsilon=0.3),
    #LIN:svm.LinearSVR(C=0.009, epsilon=0.4),
    Rand_F:ensemble.RandomForestRegressor(n_estimators=50),
    GPR:gaussian_process.GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True),
    SGD:linear_model.SGDRegressor(),
    KRR:kernel_ridge.KernelRidge(kernel='linear', alpha=(1/20))
}

def get_model_fixed(model_type):
    return MODELS_DIC[model_type]

def get_model(model_type, c=0, epsilon=0, gamma=0):

    if model_type == RBF:
        model =  model = svm.SVR(kernel='rbf', C=c, epsilon=epsilon, gamma=gamma)
    elif model_type == POLY2:
        model = svm.SVR(kernel='poly', C=c, degree=2, epsilon=epsilon)
    elif model_type == POLY3:
        model = svm.SVR(kernel='poly', C=c, degree=3, epsilon=epsilon)
    elif model_type == POLY4:
        model = svm.SVR(kernel='poly', C=c, degree=4, epsilon=epsilon)
    elif model_type == LIN:
        model = svm.SVR(kernel='linear', C=c, epsilon=epsilon)
    elif model_type == Rand_F:
        model = ensemble.RandomForestRegressor()
    elif model_type == SGD:
        model = linear_model.SGDRegressor()
    elif model_type == KRR:
        model = kernel_ridge.KernelRidge(kernel='linear', alpha=1/(2*c))
    elif model_type == DT:
        model = DecisionTreeRegressor()
    else:
        raise(ValueError('unknown model type: ' + str(model_type)))
    return model

def get_model_random(model_type):

    if model_type == LIN:
        c = 0.1 * 10 ** (np.random.rand() * 3)
        epsilon = np.random.rand() * 0.4
        model = svm.SVR(kernel='linear', C=c, epsilon=epsilon)
    else:
        raise(ValueError('unknown model type: ' + str(model_type)))
    return model
