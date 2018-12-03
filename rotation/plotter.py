import matplotlib.pyplot as plt
import numpy as np

def plot_one(data, actual, predicted=None):
    data = data.squeeze()
    dims = data.shape[1]
    legend = ['original', 'actual rotated']
 
    plt.scatter(data[:,0], data[:,1])
    
    rotated = np.dot(data, actual)
    plt.scatter(rotated[:,0], rotated[:,1])
    
    if predicted is not None:
        predicted = predicted.squeeze()
        pred_rot = np.dot(data, predicted)
        plt.scatter(pred_rot[:,0], pred_rot[:,1])
        legend.append('predicted rotated')
    
    plt.legend(legend)
