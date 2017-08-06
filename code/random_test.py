import numpy as np
import PlotHelper as ph

correlations = np.zeros([10000,1])

for i in range(10000):
    rand1 = np.random.normal(size=[100, 1])
    rand2 = np.random.normal(size=[100, 1])
    correlations[i,0] = np.corrcoef(rand1.ravel(), rand2.ravel())[1,0]

    #print('iter ' + str(i) + ' correlation: ' + str(np.corrcoef(rand1.ravel(), rand2.ravel())[1,0]))

title = 'Correlation Between Independent Normally Distributed Random Variables'
ph.draw_hist(correlations, title, 'Correlation Coefficient')