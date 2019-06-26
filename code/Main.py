from timeit import default_timer as timer

import dateutil
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import code.Valid as valid

import code.DataHelper as dh
import code.ModelHelper as mh
import code.PlotHelper as ph
from code.EnsembleSVR import EnsembleSVR
from sklearn.ensemble import BaggingRegressor

start = timer()
LOAD_FROM_FILE = False
LOAD_DELTAS = False
MODEL_TYPE = mh.LIN

FORECAST_QUARTERS = 1

DO_VALIDATION = True
DO_FUTURE_FORECAST = False
DO_TEST = True

TEST_ON_TRAIN = False #test mode
EXPANDING_WINDOW = True
ROLLING_WINDOW = False
USE_SEASONS = False
USE_ALL_EXPERTS = False
USE_ENSEMBLE = False
ENSEMBLE_EQUALLY_WEIGHTED = True
EXAMINE_RESIDUALS = False
ANALYSE_VARIANCE = False

START_TEST_DATE = dateutil.parser.parse("2007-01-01")
target_col = 'PCECC96'


def residuals_correlation_score(y_true, y_pred):
    residuals  = y_true - y_pred
    correlation = np.corrcoef(residuals, y_pred)
    score = abs(correlation[0, 1]) + sum(abs(y_true - y_pred)) * 0.01
    return score

def run_tests(model, data, results):

    print('Starting Test!')
    if ANALYSE_VARIANCE:
        test_results = pd.DataFrame()
        for i in range(10):
            model = mh.get_model_random(mh.LIN)
            if not USE_ENSEMBLE:
                ensemble_indices = np.random.choice(17, size=10, replace=False)
                data_for_test = data.iloc[:, ensemble_indices]
            else:
                data_for_test = data

            all_predictions, true_values_test, dates_test, experts = do_test(model, data_for_test, results)
            #ph.plot_stds(dates_test)
            temp_df = pd.DataFrame(index=dates_test, data=all_predictions, columns=[i])
            test_results = pd.concat((test_results, temp_df), axis=1)
        title = 'Dispersion of Predictions of Ensemble Linear SVR'
        ph.plot_quartiles(test_results.values,dates_test,title,true_values_test=true_values_test, all_predictions=np.median(test_results.values, axis=1), experts=experts)
    else:
        do_test(model, data, results)

def do_fwd_prediction(model, data, results):

    base_model = model
    if USE_ENSEMBLE:
        model = EnsembleSVR(MODEL_TYPE)

    for fwd_index in range(1, FORECAST_QUARTERS+1):

        num_data_items = data.shape[0]
        x_train = data[0:-fwd_index]  # data.iloc[1:num_train,2:]
        y_train = results[fwd_index:].values.ravel()
        x_test = data.iloc[-1,:].values.reshape([1,17])  # data.iloc[num_train + 1:,2:]

        #model.reweight(x_train, y_train)

        # normalise columns
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        print(prediction)


def do_test(model, data, results):

    base_model = model
    if USE_ENSEMBLE:
        model = EnsembleSVR(MODEL_TYPE)

    for fwd_index in range(1, FORECAST_QUARTERS+1):

        start_train = 1
        data, results = dh.shift_data_one_quarter(data, results)
        num_data_items = data.shape[0]

        if num_data_items != results.shape[0]:
            raise ValueError('Number of items in data does not match number of items in target!')

        if TEST_ON_TRAIN:
            num_test_items = 1
        else:
            num_test_items = data[data.index >= START_TEST_DATE].shape[0]

        end_train = num_data_items - num_test_items - 1
        start_test = end_train + 1

        dates_train = data.index[start_train:end_train]
        dates_test = data.index[start_test:]

        all_predictions = np.zeros([num_test_items])
        all_std = np.zeros([num_test_items])
        true_values_test = results[start_test:].values.ravel()

        for j in range(num_test_items):

            # expanding window
            if EXPANDING_WINDOW:
                end_train = num_data_items - num_test_items - 1 + j
                start_test = end_train + 1
                end_test = start_test + 1
            elif ROLLING_WINDOW:
                start_train += 1
                end_train = num_data_items - num_test_items - 1 + j
                start_test = end_train + 1
                end_test = start_test + 1

            x_train = data[start_train:end_train]  # data.iloc[1:num_train,2:]
            y_train = results[start_train:end_train].values.ravel()
            x_test = data[start_test:end_test]  # data.iloc[num_train + 1:,2:]

            if USE_ENSEMBLE and not ENSEMBLE_EQUALLY_WEIGHTED and j == 0:
                model.reweight(x_train, y_train)

            # normalise columns
            scaler = preprocessing.StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            if TEST_ON_TRAIN:
                x_test = x_train
                dates_test = dates_train
                true_values_test = y_train

            model.fit(x_train, y_train)
            # temp = model.coef_
            # temp2 = model.dual_coef_
            prediction = model.predict(x_test)
            if TEST_ON_TRAIN:
                all_predictions = prediction
            else:
                all_predictions[j] = prediction


        end_date = dates_test[-1]
        experts = dh.get_experts(LOAD_FROM_FILE)
        expert_col = experts['fwd' + str(fwd_index)]
        experts_test = expert_col[(expert_col.index >= START_TEST_DATE) & (expert_col.index <= end_date)]

        if ANALYSE_VARIANCE:
            return all_predictions, true_values_test, dates_test, experts_test
        else:
            process_predictions(all_predictions, true_values_test, dates_test, base_model, fwd_index, experts_test)



def process_predictions(all_predictions, true_values_test, dates_test, model, fwd_index, experts_test):

    end_date = dates_test[-1]
    if USE_ALL_EXPERTS:
        experts = dh.get_all_experts(LOAD_FROM_FILE)
        expert_col = experts[fwd_index - 1]
        experts_test = expert_col[(expert_col.index >= START_TEST_DATE) & (expert_col.index <= end_date)]
        title = str(fwd_index) + ' periods forward, Model=Ensemble SVR, KERNEL=' + model.get_params()['kernel']
        ph.plot_quartiles(experts_test,dates_test,title,true_values_test=true_values_test,all_predictions=all_predictions)
        ph.plot_percentage_beaten(experts_test,dates_test,true_values_test,all_predictions)

    mse = metrics.mean_squared_error(true_values_test, all_predictions)
    expert_mse = metrics.mean_squared_error(true_values_test, experts_test)

    mae = metrics.mean_absolute_error(true_values_test, all_predictions)
    expert_mae = metrics.mean_absolute_error(true_values_test, experts_test)

    hinge_loss = str(round(dh.hinge_loss(true_values_test, all_predictions, 0.5), 4))
    expert_hinge_loss = str(round(dh.hinge_loss(true_values_test, experts_test, 0.5), 4))

    right_direction_score = dh.right_direction_score(true_values_test, all_predictions)
    expert_right_direction_score = dh.right_direction_score(true_values_test, experts_test)

    r_squared = metrics.r2_score(true_values_test, all_predictions)
    expert_r_squared = metrics.r2_score(true_values_test, experts_test)

    # print(np.var(all_predictions))
    # print(np.var(experts_test))

    correlation = np.corrcoef(true_values_test, all_predictions)
    expert_correlation = np.corrcoef(true_values_test, experts_test)

    if EXAMINE_RESIDUALS:
        residuals = true_values_test - all_predictions
        title = str(fwd_index) + ' periods forward residuals, correlation = ' + str(round(np.corrcoef(all_predictions, residuals)[0, 1],4))
        ph.plot_one_scatter(all_predictions, residuals, title, fwd_index + FORECAST_QUARTERS * 2, 'Predicted CS Growth', 'True Value - Prediction')

    if MODEL_TYPE == mh.RBF:
        gamma_string = str(round(model.get_params()['gamma'],5))
    else:
        gamma_string = ' - '

    if MODEL_TYPE == mh.Rand_F or MODEL_TYPE == mh.DT:
        epsilon_string = ' - '
        C_string = ' - '
        model_string = 'Random Forest'
    else:
        epsilon_string = str(round(model.get_params()['epsilon'],4))
        C_string = str(round(model.get_params()['C'],4))
        model_string = model.get_params()['kernel'] + ' SVR'

    title = str(fwd_index) + ' periods forward, Model=' + model_string # KERNEL=' + 'Linear' # model.get_params()['kernel']) #SVR, Kernel = POLY3')
    ph.plot_forecast_performance(fwd_index, all_predictions, true_values_test, experts_test, dates_test, title)

    # print(str(i) + ' & Mean Experts & - & - & - & ' + str(round(expert_mse,4)) + ' & ' + str(round(expert_mae,4)) + ' & ' + expert_hinge_loss + ' & '
    #             + str(round(expert_correlation[0,1],4)) + ' & ' + str(round(expert_r_squared,4)))
    #print(str(round(correlation[0,1],4)))

    print(str(fwd_index) + ' & ' + model_string + ' & ' + C_string +
          ' & ' + epsilon_string + ' & ' + gamma_string +
          ' & ' + str(round(mse,4)) + ' & ' + str(round(mae,4)) + ' & ' + hinge_loss + ' & '
                + str(round(correlation[0,1],4)) + ' & ' + str(round(r_squared,4)))


def main():

    import os
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/data/'

    #data_file = data_dir + 'input_data_full.csv'
    #data_file = data_dir + 'InputData_small.csv'
    #results_file = data_dir + 'consumer_spending.csv'
    #results_file = data_dir + 'Fake_Results.csv'
    data = dh.get_all_data(LOAD_FROM_FILE, LOAD_DELTAS, data_dir)
    results = data[target_col]

    num_data_items = data.shape[0]
    season_info = np.zeros([num_data_items,4])
    j = 0
    for i in range(num_data_items):
        season_info[i,j] = 1
        j += 1
        if j > 3:
            j = 0

    seasons_df = pd.DataFrame(data=season_info, columns=['SEASON_1', 'SEASON_2', 'SEASON_3', 'SEASON_4'])
    seasons_df.index = data.index

    if USE_SEASONS:
        data = pd.concat([data, seasons_df], axis=1)

    if DO_VALIDATION:
        model = valid.do_validation(data, results, MODEL_TYPE, USE_ENSEMBLE)
        if DO_TEST:
            run_tests(model, data, results)
    elif DO_FUTURE_FORECAST:
        model = mh.get_model_fixed(MODEL_TYPE)
        do_fwd_prediction(model,data,results)
    else:
        model = mh.get_model_fixed(MODEL_TYPE)
        #do_fwd_prediction(model,data,results)
        run_tests(model, data, results)

    end = timer()
    print('finished in: ' + str(end - start))
    ph.show_plots()



main()