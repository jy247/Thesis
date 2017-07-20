import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
from timeit import default_timer as timer

start = timer()
FORECAST_QUARTERS = 2

data_dir = 'C:/Users/Jack/Documents/Thesis/data/'
data_file = data_dir + 'input_data_full.csv'
#data_file = data_dir + 'InputData_small.csv'
results_file = data_dir + 'consumer_spending.csv'
#results_file = data_dir + 'Fake_Results.csv'

data = pandas.read_csv(data_file)
results = pandas.read_csv(results_file)
num_data_items = data.shape[0]
print(data)
num_columns = data.shape[1]
titles = data.columns
dates = data.iloc[FORECAST_QUARTERS:,0]
data = data.iloc[:num_data_items - FORECAST_QUARTERS,1:num_columns-1]
results = results.iloc[FORECAST_QUARTERS:,0].ravel()

num_columns = data.shape[1]
for i in range(0, 1):
    one_factor = data.iloc[:,i]

    #normalise columns
    scaler = preprocessing.StandardScaler().fit(one_factor)
    scaled_factor = scaler.transform(one_factor)

    plt.figure(i)
    # plt.plot(dates, one_factor, label='factor')
    # plt.plot(dates, scaled_factor, label='scaled factor')
    # plt.plot(dates, results, label='results')
    plt.scatter(one_factor, results)
    plt.xlabel('date')
    plt.ylabel('growth %')
    plt.legend()
    correlation = np.corrcoef(results, scaled_factor)
    plt.title(str(titles[i]) + ': correlation = ' + str(correlation[0,1]))
    print('iteration: ' + str(i) + ' completed')

end = timer()
print('finished in: ' + str(end - start))
plt.show()

