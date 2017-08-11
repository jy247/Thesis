import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


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