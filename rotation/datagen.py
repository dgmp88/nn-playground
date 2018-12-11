import keras
import numpy as np

def make_rotmat(rad):
    return np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]
    ])


def make_one(n_pts, dims, noise):
    # Generate random points 
    data = np.random.rand(n_pts, dims)
    
    # Make a rotation matrix
    rad = np.random.uniform(0, np.pi*2)
    rotmat = make_rotmat(rad)
    rotated = np.dot(data, rotmat)
        
    # Make some noise 
    if noise:
        rotated += np.random.uniform(low=-noise, high=noise, size=rotated.shape)
    
    # Invert the rotation so we get back frmo the 2nd to 1st set of points
    rotmat = rotmat.T
    
    return data, rotated, rotmat

class DataGenerator(keras.utils.Sequence):
    def __init__(self, dims, n_pts, batch_size, steps_per_epoch, noise=0):
        self.dims = dims
        self.n_pts = n_pts
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.noise = noise

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        X = np.empty((self.batch_size, self.n_pts, self.dims*2))
        y = np.empty((self.batch_size, self.dims, self.dims))
        for idx in range(self.batch_size):
            a, b, r = make_one(self.n_pts, self.dims, self.noise)
            X[idx, :, 0:2] = a
            X[idx, :, 2:4] = b
            y[idx, :, :] = r

        return X, y
    
    def get_one(self):
        data, x = self[np.random.choice(len(self))]

        data, x = data[[0]], x[0]
        return data, x
