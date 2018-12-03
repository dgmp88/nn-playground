import keras
import numpy as np


def make_one(n_pts, dims):
    data = np.random.rand(n_pts, dims)

    rotmat = np.eye(dims, dims)
    rotmat = (rotmat==0) * np.random.uniform(low=0, high=1, size=(dims, dims))

    rotated = np.dot(data, rotmat)
    
    return data, rotated, rotmat

class DataGenerator(keras.utils.Sequence):
    def __init__(self, dims, n_pts, batch_size, steps_per_epoch):
        self.dims = dims
        self.n_pts = n_pts
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        X = np.empty((self.batch_size, self.n_pts, self.dims*2))
        y = np.empty((self.batch_size, self.dims, self.dims))
        for idx in range(self.batch_size):
            a, b, r = make_one(self.n_pts, self.dims)
            X[idx, :, 0:2] = a
            X[idx, :, 2:4] = b
            y[idx, :, :] = r

        return X, y
    
    def get_one(self):
        data, x = self[np.random.choice(len(self))]

        data, x = data[[0]], x[0]
        return data, x
