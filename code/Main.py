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
import DataHelper as dh
import ModelHelper as mh
import PlotHelper as ph
import dateutil

start = timer()
load_from_file = False
MODEL_TYPE = mh.LIN
FORECAST_QUARTERS = 4
DO_X_Valid = True
DO_TEST = True
X_VALID_QUARTERS = 1
EXPORT_X_VALID = False
#X_VALID_SCORE_MODEL = 'neg_median_absolute_error'
#X_VALID_SCORE_MODEL = 'neg_mean_squared_error'
X_VALID_SCORE_MODEL = 'r2'

TEST_ON_TRAIN = False #test mode
USE_SELECT_COLUMNS = False
EXPANDING_WINDOW = True
ROLLING_WINDOW = True
USE_SEASONS = False
USE_GPR = False

examine_residuals = False
START_TEST_DATE = dateutil.parser.parse("2007-01-01")
target_col = 'PCECC96'

def get_gamma_range():
    if MODEL_TYPE == mh.RBF:
        return 6
    else:
        return 1

def shift_data_one_quarter(data, results):
    # process for forecasting
    data = data[:-1]
    results = results.shift(-1)
    results = results[:-1]
    return data, results

def do_x_validation(data, results):

    best_sum_score = float('-inf')
    best_c = 0
    best_epsilon = 0
    best_gamma = 0
    all_c = []
    all_epsilon = []
    all_score = []
    all_gamma = []
    for i in range(X_VALID_QUARTERS):
        data, results = shift_data_one_quarter(data, results)

    for precision_loop in range(2):

        if precision_loop == 1:
            print('Do more precise grid search')
            # more precise
            gamma_increment = best_gamma /10
            epsilon_increment = best_epsilon / 10
            c_increment = best_c / 10
            low_gamma = best_gamma - (3 * gamma_increment)
            low_epsilon = best_epsilon - (8 * epsilon_increment)
            low_c = best_c - (3 * c_increment)

        for c_factor in range(6):
            for epsilon_factor in range(16):
                for gamma_factor in range(get_gamma_range()):

                    if precision_loop == 0:
                        gamma = 0.0001 * 10 ** gamma_factor
                        epsilon = 0.1 * epsilon_factor
                        c = 0.001 * 10 ** c_factor
                    else:
                        # more precise
                        gamma = low_gamma + gamma_increment * gamma_factor
                        epsilon = low_epsilon + epsilon_increment * epsilon_factor
                        c = low_c + c_increment * c_factor

                    model = mh.get_model(MODEL_TYPE, c, epsilon, gamma)
                    pipeline = make_pipeline(preprocessing.StandardScaler(), model)
                    scores = cross_val_score(pipeline, data, results, cv=5, scoring=X_VALID_SCORE_MODEL)
                    all_c.append(c)
                    all_epsilon.append(epsilon)
                    all_gamma.append(gamma)
                    all_score.append(sum(scores))
                    if sum(scores) > best_sum_score:
                        print(scores)
                        best_sum_score = sum(scores)
                        best_c = c
                        best_epsilon = epsilon
                        best_gamma = gamma

            print('iter: ' + str(c_factor))
        #ph.plot_surf(all_c, all_epsilon, all_score, 1)
        if EXPORT_X_VALID and precision_loop == 0:
            try:
                df = pd.read_csv('xvalid_analysis3.csv')
            except:
                df = pd.DataFrame()
                df['c'] = all_c
                df['epsilon'] = all_epsilon
                df['gamma'] = all_gamma
            df['score' + X_VALID_SCORE_MODEL + ' ' + str(X_VALID_QUARTERS)] = all_score
            df.to_csv('xvalid_analysis3.csv', index=False)


    print(model.get_params()['kernel'] + ' & ' + X_VALID_SCORE_MODEL + ' & ' + str(round(best_c,3)) + ' & ' + str(round(best_epsilon,3))
          + ' & ' + str(round(best_gamma,4)) + ' & ' + str(round(best_sum_score,3)) )
    return mh.get_model(MODEL_TYPE, best_c, best_epsilon, best_gamma)

def do_test(model, data, results, experts):

    print('Starting Test!')
    for i in range(FORECAST_QUARTERS):

        start_train = 1
        data, results = shift_data_one_quarter(data, results)
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

            # normalise columns
            scaler = preprocessing.StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            if TEST_ON_TRAIN:
                x_test = x_train
                dates_test = dates_train
                true_values_test = y_train

            # print('iteration: ' + str(i) + ' fitting')
            if MODEL_TYPE == mh.gaussian_process:
                model.fit(x_train, y_train)
                y_gpr, y_std = model.predict(x_test, return_std=True)
                all_predictions[j] = y_gpr
                all_std[j] = y_std
            else:
                model.fit(x_train, y_train)
                # print('iteration: ' + str(i) + ' fitted!')
                # temp = model.coef_
                # temp2 = model.dual_coef_
                prediction = model.predict(x_test)
                if TEST_ON_TRAIN:
                    all_predictions = prediction
                else:
                    all_predictions[j] = prediction

        process_predictions(all_predictions, true_values_test, dates_test, model, i+1, experts, all_std)

def process_predictions(all_predictions, true_values_test, dates_test, model, i, experts, all_std):

    plt.figure(i)
    end_date = dates_test[-1]
    # plt.scatter(predictions, y_test)
    # plt.xlabel('predictions')
    # plt.ylabel('actual values')

    expert_col = experts['fwd' + str(i)]
    experts_test = expert_col[(expert_col.index >= START_TEST_DATE) & (expert_col.index <= end_date)]

    # cheat by adjusting the mean and std of the predictions
    # mean_true = np.mean(true_values_test)
    # mean_predicted = np.mean(all_predictions)
    # all_predictions = all_predictions - mean_predicted + mean_true
    # sd_true = np.std(true_values_test)
    # sd_predicted = np.std(all_predictions)
    # all_predictions = all_predictions * sd_true / sd_predicted

    # mean_experts = np.mean(experts_test.values)
    # experts_test = experts_test - mean_experts + mean_true
    # sd_experts = np.std(experts_test.values)
    # experts_test = experts_test * sd_true / sd_experts

    plt.plot(dates_test, experts_test, label='experts')
    plt.plot(dates_test, all_predictions, label='predictions')
    plt.plot(dates_test, true_values_test, label='actual')

    if USE_GPR:
        lower = all_predictions - all_std
        upper = all_predictions + all_std
        plt.fill_between(dates_test, lower, upper, color='darkorange', alpha=0.2)

    plt.xlabel('date')
    plt.ylabel('growth %')
    plt.legend()

    mse = metrics.mean_squared_error(true_values_test, all_predictions)
    mae = metrics.mean_absolute_error(true_values_test, all_predictions)
    expert_mse = metrics.mean_squared_error(true_values_test, experts_test)

    try:
        epsilon = model.get_params()['epsilon']
        hinge_loss = dh.hinge_loss(true_values_test, all_predictions, epsilon)
        expert_hinge_loss = dh.hinge_loss(true_values_test, experts_test.values, epsilon)
        hinge_loss = str(round(hinge_loss, 4))
    except:
        hinge_loss = ' - '

    right_direction_score = dh.right_direction_score(true_values_test, all_predictions)
    expert_right_direction_score = dh.right_direction_score(true_values_test, experts_test.values)

    r_squared = metrics.r2_score(true_values_test, all_predictions)
    expert_r_squared = metrics.r2_score(true_values_test, experts_test.values)

    correlation = np.corrcoef(true_values_test, all_predictions)
    plt.title(str(i) + ' periods forward, correlation = '
              + str(round(correlation[0, 1],4)) + '\n score = ' + str(round(r_squared,4))
              + ' ex_score = ' + str(round(expert_r_squared,4)))

    title = str(i) + ' periods forward predictions, correlation = ' + str(round(np.corrcoef(all_predictions, true_values_test)[0, 1],4))

    ph.plot_one_scatter(true_values_test, all_predictions, title, i + FORECAST_QUARTERS)

    if examine_residuals:
        residuals = true_values_test - all_predictions
        title = str(i) + ' periods forward residuals, correlation = ' + str(round(np.corrcoef(all_predictions, residuals)[0, 1],4))
        ph.plot_one_scatter(all_predictions, residuals, title, i + FORECAST_QUARTERS * 2)

    if MODEL_TYPE == mh.RBF:
        gamma_string = str(round(model.get_params()['gamma'],5))
    else:
        gamma_string = ' - '
    if MODEL_TYPE == mh.Rand_F:
        epsilon_string = ' - '
        C_string = ' - '
        model_string = 'Random Forest'
    else:
        epsilon_string = str(round(model.get_params()['epsilon'],4))
        C_string = str(round(model.get_params()['C'],4))
        model_string = model.get_params()['kernel'] + ' SVR'

    print(str(i) + ' & ' + model_string + ' & ' + epsilon_string +
          ' & ' + C_string + ' & ' + gamma_string +
          ' & ' + str(round(mse,4)) + ' & ' + str(round(mae,4)) + ' & ' + hinge_loss + ' & '
                + str(round(correlation[0,1],4)) + ' & ' + str(round(r_squared,4)))

def main():

    data_dir = 'C:/Users/Jack/Documents/PycharmProjects/Thesis/data/'
    #data_file = data_dir + 'input_data_full.csv'
    #data_file = data_dir + 'InputData_small.csv'
    #results_file = data_dir + 'consumer_spending.csv'
    experts_file = data_dir + 'experts_out.csv'
    #results_file = data_dir + 'Fake_Results.csv'

    experts = dh.get_experts(load_from_file)
    data = dh.get_all_data(load_from_file, True)
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

    columns_to_use = ['PCECC96', 'TWEXM', 'HOUST', 'CIVPART','POP','UNRATE','PSAVERT','W875RX1','TOTALSA', 'M2','DGS10',
                      'UMCSENT', 'USTRADE','CPIAUCSL','WTISPLC','WILL5000INDFC',
                      'TWEXM_delta', 'TOTALSL_delta', 'POP_delta',
                      'PSAVERT_delta', 'M2_delta', 'DGS10_delta', 'UMCSENT_delta', 'CPIAUCSL_delta', 'WILL5000INDFC_delta']

    #columns_to_use = ['HOUST', 'USTRADE', 'DGS10', 'TOTALSL', 'UMCSENT', 'PSAVERT', 'WTISPLC', 'M2', 'WILL5000INDFC']

    # dates = data.iloc[:,0]
    # dates = pd.date_range(pd.datetime(1950,4,1), periods=num_data_items,freq='3M')

    if USE_SELECT_COLUMNS:
        data = data[columns_to_use]

    if USE_SEASONS:
        data = pd.concat([data, seasons_df], axis=1)

    if DO_X_Valid:
        model = do_x_validation(data, results)
        if DO_TEST:
            do_test(model, data, results, experts)
    else:
        model = mh.get_model_fixed(MODEL_TYPE)
        do_test(model, data, results, experts)

    end = timer()
    print('finished in: ' + str(end - start))
    plt.show()

main()