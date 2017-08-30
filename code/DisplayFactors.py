import numpy as np
import dateutil
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from sklearn import preprocessing

import DataHelper as dh
import PlotHelper as ph

#import center_spines as cs

START_TEST_DATE = dateutil.parser.parse("2007-01-01")
SHOW_GRAPHS = False
LOAD_FROM_FILE = True

def get_corr_for_one(data, results, titles, i):
    one_factor = data.iloc[:,i]
    one_factor = np.reshape(one_factor.values, [1, -1])[0]
    one_factor = one_factor.reshape(-1,1)

    #normalise columns
    scaler = preprocessing.StandardScaler().fit(one_factor)
    scaled_factor = scaler.transform(one_factor)
    correlation = np.corrcoef(results.ravel(), scaled_factor.ravel())[0,1]
    tmp = linregress(results.ravel(), scaled_factor.ravel())
    print(tmp)
    # FRED Key & Series Name & Corr 1 fwd  & Corr 4 fwd & Delta Corr 1 fwd & Delta Corr 4 fwd \\
    if SHOW_GRAPHS:
        if abs(correlation) > 0.18:
            title = str(titles[i]) + ': correlation = ' + str(correlation)
            ph.plot_one(scaled_factor, results, title, i)
    return str(np.round(correlation,3))

def check_one_forecast_gap(data, forecast_quarters, use_deltas):

    results = data['PCECC96'].shift(-forecast_quarters)
    titles = data.columns
    data = data[:-forecast_quarters]
    results = results[:-forecast_quarters]
    # data['cs_shifted'] = results
    # data.to_csv('../data/cs_compare.csv')
    results = results.reshape(-1, 1)
    num_columns = data.shape[1]
    if use_deltas:
        num_orig_columns = int(num_columns / 2)
    else:
        num_orig_columns = int(num_columns)

    delta_correlations = np.zeros([num_orig_columns, 1])
    correlations = np.zeros([num_orig_columns, 1])

    for i in range(num_columns):
        if i >= num_orig_columns:
            delta_correlations[i - num_orig_columns, 0] = get_corr_for_one(data, results, titles, i)
        else:
            correlations[i, 0] = get_corr_for_one(data, results, titles, i)

    return [correlations, delta_correlations]


def DisplayFactors():

    use_deltas = True
    data = dh.get_all_data(LOAD_FROM_FILE, use_deltas)
    titles = data.columns
    data = data[data.index < START_TEST_DATE]
    check_one_forecast_gap(data, 1)
    [corr_1fwd,corr_1fwd_delta] = check_one_forecast_gap(data, 1, use_deltas)
    [corr_4fwd, corr_4fwd_delta] = check_one_forecast_gap(data, 4, use_deltas)

    for i in range(corr_1fwd.shape[0]):
        if use_deltas:
            print(titles[i] + ' & pch & ' + str(round(corr_1fwd[i,0],4)) + ' &  ' + str(round(corr_1fwd_delta[i,0],4)) + ' &  '
                        + str(round(corr_4fwd[i,0],4)) + ' & ' + str(round(corr_4fwd_delta[i,0],4)) + ' \\\\')
        else:
            print(titles[i] + ' & pch & ' + str(round(corr_1fwd[i, 0],4)) + ' &  ' + str(round(corr_4fwd[i, 0],4)) +  '\\\\')
    ph.draw_hist(corr_1fwd, '1 Period Forward Correlations Between Input Variables and Target', 'Correlation Coefficient')
    plt.show()

DisplayFactors()