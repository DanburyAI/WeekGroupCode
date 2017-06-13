import numpy as np

class DataReader(object):
    def __init__(self):
        N = 1000
        theta = np.linspace(0, 2*np.pi + (np.pi / 6.), N)
        r = 0.5 + 2*theta
        y0 = r*np.cos(theta) + np.random.normal(scale=0.5, size=(N,))
        y1 = r*np.sin(theta) + np.random.normal(scale=0.5, size=(N,))
        z = np.random.uniform(size=(N,))
        self.y = np.array(zip(y0,y1))
        self.z = z

class BatchDataset(object):
    def __init__(self, y, z):
        self.Y = y
        self.Z = z
        r,c = y.shape
        self.N = r # number of samples
        self.D = c # dimension of data
        self.batch_offset = 0
        self.epochs_completed = 0

    def get_records(self):
        return self.Y, self.Z

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.Y.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.Y.shape[0])
            np.random.shuffle(perm)
            self.Y = self.Y[perm]
            self.Z = self.Z[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        return self.Y[start:end], self.Z[start:end]
