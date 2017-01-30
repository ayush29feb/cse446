import numpy as np

def logistic_loss(X, y, w):
    N = X.shape[0] # N is expected to be 1 in sgd
    y_ = (1 + np.exp(-1 * np.matmul(X, w))) ** -1 # X (N, D) w(D,) = Xw (N,)
    if y is None:
        return y_

    loss = np.sum((y - (y_ > 0.5) * 1) ** 2) / N
    dw = np.matmul(X.T, y - y_)
    return (loss, dw)