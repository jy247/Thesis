import matplotlib.pyplot as plt
import numpy as np

def examine_residuals(predictions, actual,i):

    residuals = actual - predictions
    plt.figure(i+4)
    plt.scatter(predictions, residuals)
    plt.xlabel('predictions')
    plt.ylabel('residuals')
    plt.title(str(i + 1) + ' periods forward, correlation = ' + str(np.corrcoef(predictions,residuals)[0,1]))


    plt.figure(i+8)
    plt.hist(residuals, bins=15, histtype='step')
    plt.title(str(i + 1) + ' periods forward, average = ' + str(np.mean(residuals)))

