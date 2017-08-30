from sklearn import metrics
from sklearn import preprocessing
import numpy as np
import DataHelper as dh
import ModelHelper as mh
import pandas as pd
import EnsembleSVR
import PlotHelper as ph

VALID_QUARTERS = 1
EXPORT_VALID = True
VALID_SCORE_MODEL = 'mae'

def get_gamma_range(model_type):
    if model_type == mh.RBF:
        return 6
    else:
        return 1

def get_validation_score(x_train, y_train, model):
    n_train = x_train.shape[0]
    n_valid = int(n_train / 4)
    n_train = n_train - n_valid

    x_valid = x_train[n_train:]
    x_train = x_train[0:n_train-1]
    y_valid = y_train[n_train:]
    y_train = y_train[0:n_train-1]

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)

    model.fit(x_train, y_train)
    predictions = model.predict(x_valid)
    if VALID_SCORE_MODEL == 'corr':
        score = np.corrcoef(y_valid, predictions)[0,1]
    elif VALID_SCORE_MODEL == 'r2':
        score = metrics.r2_score(y_valid, predictions)
        if score > 0.5:
            ph.plot_forecast_performance(1,predictions,y_valid,y_valid,range(n_valid),'test')
            ph.show_plots()
    elif VALID_SCORE_MODEL == 'mse':
        score = -metrics.mean_squared_error(y_valid, predictions)
    elif VALID_SCORE_MODEL == 'mae':
        score = -metrics.mean_absolute_error(y_valid, predictions)
    elif VALID_SCORE_MODEL == 'hinge':
        score = -dh.hinge_loss(y_valid, predictions, 0.25)
    elif VALID_SCORE_MODEL == 'joint':
        score = np.corrcoef(y_valid, predictions)[0, 1] - dh.hinge_loss(y_valid, predictions, 0.25)
    elif VALID_SCORE_MODEL == 'median':
        score = -np.median(np.abs(y_valid - predictions))
    else:
        raise (ValueError('unknown VALID_SCORE_MODEL: ' + str(VALID_SCORE_MODEL)))

    return score

def do_validation(data, results, model_type, use_ensemble):

    best_sum_score = float('-inf')
    best_c = 0
    best_epsilon = 0
    best_gamma = 0
    all_c = []
    all_epsilon = []
    all_score = []
    all_gamma = []
    #residuals_correlation_scorer = make_scorer(residuals_correlation_score)
    for i in range(VALID_QUARTERS):
        data, results = dh.shift_data_one_quarter(data, results)

    for precision_loop in range(2):

        if precision_loop == 1:
            print('Do more precise grid search')
            # more precise
            epsilon_increment = best_epsilon / 10
            low_gamma = best_gamma / 8
            low_epsilon = best_epsilon - (5 * epsilon_increment)
            low_c = best_c / 8

        for c_factor in range(1):
            for epsilon_factor in range(10):
                for gamma_factor in range(get_gamma_range(model_type)):

                    if precision_loop == 0:
                        gamma = 0.0001 * 10 ** gamma_factor
                        epsilon = 0.1 * epsilon_factor
                        c = 100 * 10 ** c_factor
                    else:
                        # more precise
                        gamma = low_gamma * 2 ** gamma_factor
                        epsilon = low_epsilon + epsilon_increment * epsilon_factor
                        c = low_c * 2 ** c_factor

                    model = mh.get_model(model_type, c, epsilon, gamma)
                    if use_ensemble:
                        #model = AdaBoostRegressor(model)
                        #model = BaggingRegressor(model, max_features=10, n_estimators=20, max_samples=80)
                        model = EnsembleSVR(model_type)

                    #pipeline = make_pipeline(preprocessing.StandardScaler(), model)
                    #scores = cross_val_score(pipeline, data, results, cv=5, scoring=corr_test)
                    score = get_validation_score(data, results, model)

                    all_c.append(c)
                    all_epsilon.append(epsilon)
                    all_gamma.append(gamma)
                    all_score.append(score)
                    if score > best_sum_score:
                        print(score)
                        best_sum_score = score
                        best_c = c
                        best_epsilon = epsilon
                        best_gamma = gamma

                       # model = mh.get_model_fixed(MODEL_TYPE)
                       # pipeline = make_pipeline(preprocessing.StandardScaler(), model)
                        #score = get_validation_score(pipeline, data, results, cv=5, scoring=X_VALID_SCORE_MODEL)
                        #print(score)

            print('iter: ' + str(c_factor))
        #ph.plot_surf(all_c, all_epsilon, all_score, 1)
        if EXPORT_VALID and precision_loop == 0:
            #try:
             #   df = pd.read_csv('xvalid_analysis3.csv')
            #except:
            df = pd.DataFrame()
            df['c'] = all_c
            df['epsilon'] = all_epsilon
            df['gamma'] = all_gamma
            df['score' + VALID_SCORE_MODEL + ' ' + str(VALID_QUARTERS)] = all_score
            df.to_csv('xvalid_analysis.csv', index=False)

    if use_ensemble:
        kernel = 'RBF'  #model.models[0].get_params()['kernel']
    else:
        kernel = model.get_params()['kernel']

    print(kernel + ' & ' + VALID_SCORE_MODEL + ' & ' + str(round(best_c,3)) + ' & ' + str(round(best_epsilon,3))
              + ' & ' + str(round(best_gamma,5)) + ' & ' + str(round(best_sum_score,3)) )


    return mh.get_model(model_type, best_c, best_epsilon, best_gamma)