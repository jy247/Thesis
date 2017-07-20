import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import t

DISPLAY_DELTAS = True
DELTA_GAP = 4
start_train = 1

data_dir = 'C:/Users/Jack/Documents/Thesis/data/'

results_file = data_dir + 'consumer_spending.csv'
results = pandas.read_csv(results_file)
results = results.iloc[start_train:,0].ravel()
num_results = results.shape[0]
deltas = np.zeros([num_results-DELTA_GAP])
for i in range(num_results-DELTA_GAP):
    deltas[i] = results[i+DELTA_GAP] - results[i]

title = 'distribution of growth rates'
if DISPLAY_DELTAS:
    results = deltas
    title ='distribution of quarter to quarter change in growth rates'

min_val = min(results)
max_val = max(results)
plt.figure(1)
plt.hist(results, 30, normed=True)
plt.xlim(min_val, max_val)
plt.title(title)
mean_val = np.mean(results)
std = np.std(results)
print('mean: ' + str(mean_val))
print('std: ' + str(std))
x = np.linspace(min_val, max_val, 100)
#x_shifted = np.linspace(min_val- mean_val, max_val - mean_val, 100)
#tdof, tloc, tscale = t.fit(results)
#plt.plot(x, t.pdf(x_shifted, 0.1))
plt.plot(x, mlab.normpdf(x, mean_val, std))

plt.show()

