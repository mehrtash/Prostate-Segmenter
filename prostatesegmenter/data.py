import os
import numpy as np


class ProstateData(object):

    def load_mean_train(self):
        dirname = os.path.dirname(__file__)
        fname = 'mean_train.npy'
        datapath = os.path.join(dirname, 'data', fname)
        data = np.load(datapath)
        return data
