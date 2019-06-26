import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import code.DataHelper as dh


def draw_hist(results, title, xlabel):
    min_val = min(results)
    max_val = max(results)
    plt.figure(1)
    plt.hist(results, 20, normed=True)
    plt.xlim(min_val, max_val)

    plt.title(title)
    mean_val = np.mean(results)
    std = np.std(results)
    print('mean: ' + str(mean_val))
    print('std: ' + str(std))

    x = np.linspace(min_val, max_val, 100)

    plt.plot(x, mlab.normpdf(x, mean_val, std))
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.show()

def central_axis(ax):
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def plot_one_scatter(series1, series2, title,  plot_index, xlabel = '', ylabel = ''):

    fig = plt.figure(plot_index)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(series1, series2)
    #central_axis(ax)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    correlation = np.corrcoef(series1.ravel(), series2.ravel())[0, 1]
    plt.title(title)
    #plt.savefig('../Images/correlations/fwd_1/corr1_' + str(plot_index) + '.png')
    #('iteration: ' + str(plot_index) + ' completed')

def plot_surf(x, y, z, plot_index):

    fig = plt.figure(plot_index)
    ax = fig.add_subplot(111, projection='3d')

    # re-create the 2D-arrays
    z = np.abs(z)
    x1 = np.linspace(x[0], x[-1], len(x))
    y1 = np.linspace(y[0], y[-1], len(y))
    x2, y2 = np.meshgrid(x1, y1)
    z2 = griddata((x, y), z, (x2, y2))

    ax.plot_surface(x2, y2, z2)
    ax.set_xlabel('C')
    ax.set_ylabel('Epsilon')
    ax.set_zlabel('Cost')

    plt.show()

def plot_quartiles(data, dates_test, title, true_values_test = '', all_predictions = '', experts =''):

    labels = ['min', 'lower_quartile', 'median', 'upper_quartile', 'max' ]
    quartiles = dh.get_quartiles(data)

    fig = plt.figure(98)

    if not isinstance(true_values_test, str):
        plt.plot(dates_test, true_values_test, 'g', label='actual')
    if not isinstance(all_predictions, str):
        plt.plot(dates_test, all_predictions, 'darkorange', label='predictions')
        sparecolour = 'blue'
    if not isinstance(experts, str):
        plt.plot(dates_test, experts, 'b', label='experts')
        sparecolour = 'darkorange'

    plt.fill_between(dates_test, quartiles[1], quartiles[2], color=sparecolour, alpha=0.4)
    plt.fill_between(dates_test, quartiles[2], quartiles[3], color=sparecolour, alpha=0.4)
    plt.fill_between(dates_test, quartiles[0], quartiles[1], color=sparecolour, alpha=0.2)
    plt.fill_between(dates_test, quartiles[3], quartiles[4], color=sparecolour, alpha=0.2)

    stds = np.std(data, axis=1)
    #print(stds)
    print('mean std: ' + str(round(np.mean(stds),4)))

    plt.ylabel("growth %")
    plt.xlabel('Date')
    plt.legend()
    plt.title(title)

    plt.show()

def plot_percentage_beaten(data, dates_test, true_values_test, all_predictions):

    percentage_beaten = dh.percentage_beaten(true_values_test, all_predictions, data)
    fig = plt.figure(99)
    plt.plot(dates_test, percentage_beaten, 'r', label='percentage beaten')
    average = np.ones(dates_test.shape[0]) * np.mean(percentage_beaten)
    plt.plot(dates_test, average, 'b', label='average beaten')
    plt.ylabel("% experts")
    plt.legend()
    plt.title('Model Prediction Closer to the True Value than % of Experts')

def plot_forecast_performance(fwd_index, all_predictions, true_values_test, experts_test, dates_test, title):

    plt.figure(fwd_index)
    plt.plot(dates_test, experts_test, label='experts')
    plt.plot(dates_test, all_predictions, label='predictions')
    plt.plot(dates_test, true_values_test, label='actual')

    plt.xlabel('date')
    plt.ylabel('growth %')
    plt.legend()
    plt.title(title)

def show_plots():
    plt.show()

def plot_stds(dates_test):

    plt.figure(1)
    test_a = [ 0.1359753   ,0.11892965,  0.22531649,  0.11783498,  0.16718536 , 0.2828878,
  0.34741863 , 0.88979877  ,0.44119991 , 0.20285401,  0.28919532  ,0.15253558,
  0.16787023 , 0.11896765 , 0.0854972  , 0.11994553  ,0.19807678  ,0.17658903,
  0.26250211,  0.22429845  ,0.18496837  ,0.08781244  ,0.12331621  ,0.24659206,
  0.26337032 , 0.1550499   ,0.17389776 , 0.15365258  ,0.11085715  ,0.11130872,
  0.07658753 , 0.09519662 , 0.04991961 , 0.1381821   ,0.082499    ,0.03113242,
  0.1623467  , 0.13291557 , 0.03854866 , 0.12811952  ,0.08993153]
    test_b = [ 0.0391117  , 0.01641614,  0.03393603,  0.0279515   ,0.02933997 , 0.05504065,
  0.03010534 , 0.1210422 ,  0.06101437,  0.04522153  ,0.03805281 , 0.02492398,
  0.03088422 , 0.04007868,  0.0265191 ,  0.02544976  ,0.03316549 , 0.04921547,
  0.04725086 , 0.02345655,  0.02895945,  0.01498882  ,0.02005133 , 0.03060767,
  0.02982109 , 0.02527582  ,0.03071992,  0.02286903  ,0.02577817 , 0.02248637,
  0.01125862 , 0.02031303  ,0.02520306,  0.01952809  ,0.01918506 , 0.00867568,
  0.03251386 , 0.01276626 , 0.011634  ,  0.02306355  ,0.01214408]
    plt.plot(dates_test, test_a, label='Individual')
    plt.plot(dates_test, test_b, label='Ensemble')

    plt.xlabel('date')
    plt.ylabel('std %')
    plt.legend()
    plt.title('Standard Deviation of Forecasts Individual Vs Ensemble')
    plt.show()
