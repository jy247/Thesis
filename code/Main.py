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
from sklearn.gaussian_process import kernels
from timeit import default_timer as timer
import DataHelper as fh
import Analysis
import datetime
import dateutil


start = timer()
load_from_file = True
FORECAST_QUARTERS = 1
TEST_ON_TRAIN = False
USE_SELECT_COLUMNS = True
EXPANDING_WINDOW = False
ROLLING_WINDOW = True
USE_SEASONS = False
USE_GPR = False
examine_residuals = False
START_TEST_DATE = dateutil.parser.parse("2007-01-01")
target_col = 'PCECC96'

if USE_GPR:
    gp_kernel = kernels.DotProduct() \
                + kernels.WhiteKernel(1e-1)
    gpr = gaussian_process.GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)
else:
    #model = ensemble.RandomForestRegressor()
    #model = kernel_ridge.KernelRidge(kernel='linear',degree=2)
    #model = svm.SVR(kernel='rbf', C=1000, verbose=True)
    model = svm.SVR(kernel='linear', C=10, epsilon=0.9)
    #model = svm.SVR(kernel='poly', C=10, degree=3, epsilon=0.9)
    #model = linear_model.SGDRegressor()

data_dir = 'C:/Users/Jack/Documents/PycharmProjects/Thesis/data/'
#data_file = data_dir + 'input_data_full.csv'
#data_file = data_dir + 'InputData_small.csv'
#results_file = data_dir + 'consumer_spending.csv'
experts_file = data_dir + 'experts_out.csv'
#results_file = data_dir + 'Fake_Results.csv'

experts = fh.get_experts(load_from_file)
data = fh.get_all_data(load_from_file)
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
num_columns = data.shape[1]

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

for i in range(FORECAST_QUARTERS):

    start_train = 1
    #process for forecasting
    data = data[:-1]
    results = results.shift(-1)
    results = results[:-1]
    num_data_items = data.shape[0]

    if num_data_items != results.shape[0]:
        raise ValueError('Number of items in data does not match number of items in target!')

    num_test_items = data[data.index >= START_TEST_DATE].shape[0]
    end_train = num_data_items - num_test_items - 1
    dates_train = data.index[start_train:end_train]
    start_test = end_train + 1
    dates_test = data.index[start_test:]
    end_date = dates_test[-1]
    all_predictions = np.zeros([num_test_items])
    all_std = np.zeros([num_test_items])
    all_test = results[start_test:].values.ravel()

    for j in range(num_test_items):

        #expanding window
        if EXPANDING_WINDOW:
            end_train = num_data_items - num_test_items - 1 + j
            start_test = end_train + 1
            end_test = start_test + 1
        elif ROLLING_WINDOW:
            start_train += 1
            end_train = num_data_items - num_test_items - 1 + j
            start_test = end_train + 1
            end_test = start_test + 1

        x_train = data[start_train:end_train]#data.iloc[1:num_train,2:]
        y_train = results[start_train:end_train].values.ravel()

        #test on last 10%
        x_test = data[start_test:end_test] #data.iloc[num_train + 1:,2:]
        y_test = results[start_test:end_test].values.ravel()

        #normalise columns
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        if TEST_ON_TRAIN:
            x_test = x_train
            y_test = y_train
            dates_test = dates_train

        #print('iteration: ' + str(i) + ' fitting')
        if USE_GPR:
            gpr.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train)
        #print('iteration: ' + str(i) + ' fitted!')

       # temp = model.coef_
       # temp2 = model.dual_coef_
        if USE_GPR:
            y_gpr, y_std = gpr.predict(x_test, return_std=True)
            all_predictions[j] = y_gpr
            all_std[j] = y_std
        else:
            prediction = model.predict(x_test)
            all_predictions[j] = prediction

   # predictions = np.max([predictions, 2],)
    #for i in range(1,len(predictions)):
     #   print('regression prediction: ' + predictions[i] + ' actual: ' + y_test[i])

    plt.figure(i)
    #plt.scatter(predictions, y_test)
    #plt.xlabel('predictions')
    #plt.ylabel('actual values')
    plt.plot(dates_test, all_predictions, label='predictions')
    plt.plot(dates_test, all_test, label='actual')

    expert_col = experts['fwd' + str(i + 1)]
    experts_test = expert_col[(expert_col.index >= START_TEST_DATE) & (expert_col.index <= end_date)]
    plt.plot(dates_test, experts_test, label='experts')

    if USE_GPR:
        lower = all_predictions - all_std
        upper = all_predictions + all_std
        plt.fill_between(dates_test, lower, upper, color='darkorange', alpha=0.2)

    plt.xlabel('date')
    plt.ylabel('growth %')
    plt.legend()


    mse = metrics.mean_squared_error(all_test, all_predictions)
    expert_mse = metrics.mean_squared_error(all_test, experts_test)
    correlation = np.corrcoef(all_test, all_predictions)
    plt.title(str(i + 1) + ' periods forward, correlation = '
              + str(correlation[0,1]) + '\n MSE = ' + str(mse)
              + ' ex_MSE = ' + str(expert_mse))

    if examine_residuals:
        Analysis.examine_residuals(all_predictions, all_test,i)

    print('iteration: ' + str(i) + ' completed')

end = timer()
print('finished in: ' + str(end - start))
plt.show()

