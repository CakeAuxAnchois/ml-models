import numpy as np

def shuffle(data, labels):
    order = np.random.permutation(data.shape[0])
    return data[order], labels[order]
