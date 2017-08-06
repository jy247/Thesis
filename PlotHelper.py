import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

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

def plot_one_scatter(scaled_factor, results, titles, i, correlation):

    fig = plt.figure(i)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(scaled_factor, results)
    central_axis(ax)

    plt.title(str(titles[i]) + ': correlation = ' + str(correlation))
    plt.savefig('../Images/correlations/fwd_1/corr1_' + str(i) + '.png')
    print('iteration: ' + str(i) + ' completed')