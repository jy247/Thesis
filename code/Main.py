import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
from sklearn import kernel_ridge
from sklearn import linear_model
from timeit import default_timer as timer

start = timer()
FORECAST_QUARTERS = 4
TEST_ON_TRAIN = False
USE_SELECT_COLUMNS = True
EXPANDING_WINDOW = False
ROLLING_WINDOW = True
USE_SEASONS = False

# 1 = 1950
# 40 = 1960
# 80 = 1970
# 120 = 1980
start_train = 80

#model = ensemble.RandomForestRegressor()
#model = kernel_ridge.KernelRidge(kernel='linear',degree=2)
#model = svm.SVR(kernel='rbf', C=1000, verbose=True)
#model = svm.SVR(kernel='linear', C=10)
#model = svm.SVR(kernel='poly', C=10)
model = linear_model.SGDRegressor(loss='log')

data_dir = 'C:/Users/Jack/Documents/Thesis/data/'
data_file = data_dir + 'input_data_full.csv'
#data_file = data_dir + 'InputData_small.csv'
results_file = data_dir + 'consumer_spending.csv'
experts_file = data_dir + 'experts_out.csv'
#results_file = data_dir + 'Fake_Results.csv'

data = pd.read_csv(data_file)
results = pd.read_csv(results_file)
experts = pd.read_csv(experts_file)
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

columns_to_use = ['TWEXM', 'HOUST', 'TOTALSL', 'CIVPART','POP','UNRATE','PSAVERT','W875RX1','TOTALSA',
                  'TWEXM_delta', 'HOUST_delta', 'TOTALSL_delta', 'CIVPART_delta', 'POP_delta', 'UNRATE_delta',
                  'PSAVERT_delta', 'W875RX1_delta', 'TOTALSA_delta']

dates = data.iloc[:,0]
if USE_SELECT_COLUMNS:
    data = data[columns_to_use]
if USE_SEASONS:
    data = pd.concat([data, seasons_df], axis=1)

results = results.iloc[:,0]

for i in range(FORECAST_QUARTERS):
    #process for forecasting
    #data = data.iloc[]
    #lose the latest from data and the first from results
    dates = dates[:num_data_items - 1]
    experts = experts[:num_data_items - 1]
    data = data[:num_data_items - 1]
    results = results[1:]
    num_data_items = data.shape[0]

    if num_data_items != results.shape[0]:
        raise ValueError('Number of items in data does not match number of items in target!')

    end_train = num_data_items - 43
    dates_train = dates[start_train:end_train]
    start_test = end_train + 1
    dates_test = dates[start_test:]
    all_predictions = np.zeros([42])
    all_test = results[start_test:].values.ravel()

    #first column is t_0, then t_1 etc
    experts_test = experts.iloc[start_test:, i]

    for j in range(42):

        #expanding window
        if EXPANDING_WINDOW:
            end_train = num_data_items - 43 + j
            start_test = end_train + 1
            end_test = start_test + 1
        elif ROLLING_WINDOW:
            start_train += 1
            end_train = num_data_items - 43 + j
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
        model.fit(x_train, y_train)
        #print('iteration: ' + str(i) + ' fitted!')

       # temp = model.coef_
       # temp2 = model.dual_coef_
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
    plt.plot(dates_test, experts_test, label='experts')
    plt.xlabel('date')
    plt.ylabel('growth %')
    plt.legend()
    mse = metrics.mean_squared_error(all_test, all_predictions)
    correlation = np.corrcoef(all_test, all_predictions)
    plt.title(str(i + 1) + ' periods forward, correlation = ' + str(correlation[0,1]))
    print('iteration: ' + str(i) + ' completed')

end = timer()
print('finished in: ' + str(end - start))
plt.show()

