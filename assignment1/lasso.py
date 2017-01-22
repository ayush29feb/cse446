import numpy as np

def lasso_solver(X, y, w, reg):
    z = (X ** 2).sum(axis=0)
    while True:
        w_old = np.copy(w)
        for j in range(w.shape[0]):
            w[j] = 0
            y_ = np.matmul(X, w)
            rho_j = np.dot(X[:, j], y - y_)
            if rho_j < -reg / 2:
                w[j] = (rho_j + reg / 2) / z[j]
            elif rho_j > reg / 2:
                w[j] = (rho_j - reg / 2) / z[j]
                
        # check for convergence
        if np.max(w_old - w) < 1e-6:
            return w