import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

def plot_one(data, actual, predicted=None, s=2):
    data = data.squeeze()
    dims = data.shape[1]
    legend = ['original', 'actual rotated']
 
    plt.scatter(data[:,0], data[:,1], s=s)
    
    rotated = np.dot(data, actual)
    plt.scatter(rotated[:,0], rotated[:,1], s=s)
    
    if predicted is not None:
        predicted = predicted.squeeze()
        pred_rot = np.dot(data, predicted)
        plt.scatter(pred_rot[:,0], pred_rot[:,1], s=s)
        legend.append('predicted rotated')
    
    plt.legend(legend)