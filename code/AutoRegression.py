import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
import DataHelper as dh
import PlotHelper as ph

DISPLAY_DELTAS = False
DELTA_GAP = 1
load_from_file = False

data = dh.get_all_data(load_from_file)
results = data['PCECC96']
results = results.ravel()
num_results = results.shape[0]

title = 'Distribution of Quarterly Consumer Spending Growth Rates Since 1947'
xlabel = 'Consumer Spending Growth %'
ph.draw_hist(results, title, xlabel)


plt.show()

plt.figure(2)
shifted = results[1:]
results = results[:-1]
plt.scatter(results, shifted)
plt.xlabel('Consumer Spending Growth period t %')
plt.ylabel('Consumer Spending Growth period t+1 %')
plt.title('Auto-Correlation of Spending from One Quarter to Next')
plt.show()
print('correlation: ' + str(np.corrcoef(results, shifted)[0,1]))