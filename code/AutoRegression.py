import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
import DataHelper as dh
import PlotHelper as ph
import dateutil

DISPLAY_DELTAS = False
DELTA_GAP = 1
load_from_file = True
START_TEST_DATE = dateutil.parser.parse("2007-01-01")

data = dh.get_all_data(load_from_file, False)
results = data['PCECC96']
results = results.ravel()
num_results = results.shape[0]

#plot scatter between t and t+1
title = 'Distribution of Quarterly Consumer Spending Growth Rates Since 1947'
xlabel = 'Consumer Spending Growth %'
ph.draw_hist(results, title, xlabel)
plt.figure(2)
shifted = results[1:]
results = results[:-1]
plt.scatter(results, shifted)
plt.xlabel('Consumer Spending Growth period t %')
plt.ylabel('Consumer Spending Growth period t+1 %')
plt.title('Auto-Correlation of Spending from One Quarter to Next')
print('correlation: ' + str(np.corrcoef(results, shifted)[0,1]))

#check basic distribution of train and test sets
results = data['PCECC96']
train = results[results.index < START_TEST_DATE]
test = results[results.index >= START_TEST_DATE]

print('train mean: ' + str(np.mean(train)) + ' train std: ' + str(np.std(train)))
print('test mean: ' + str(np.mean(test)) + ' test std: ' + str(np.std(test)))
plt.figure(3)
title = 'Distribution of Quarterly Consumer Spending Train Set'
xlabel = 'Consumer Spending Growth %'
ph.draw_hist(test.ravel(), title, xlabel)

plt.figure(4)
title = 'Distribution of Quarterly Consumer Spending Test Set'
xlabel = 'Consumer Spending Growth %'
ph.draw_hist(train.ravel(), title, xlabel)