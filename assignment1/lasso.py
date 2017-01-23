import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(datadir):
    df_train = pd.read_table(datadir + '/crime-train.txt')
    df_test = pd.read_table(datadir + '/crime-test.txt')

    X_train = df_train.iloc[:, 1:].values
    y_train = df_train.iloc[:, :1].values.reshape(-1)

    X_test = df_test.iloc[:, 1:].values
    y_test = df_test.iloc[:, :1].values.reshape(-1)

    return (X_train, y_train, X_test, y_test, df_train, df_test)

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

def lasso_models(X, y, regs):
    W = np.empty([0, X.shape[1]], dtype=np.float64)
    w = np.random.normal(0, 1, W.shape[1])
    for i in range(regs.shape[0]):
        w = np.copy(W[-1]) if i != 0 else np.random.normal(0, 1, W.shape[1])
        W = np.vstack((W, lasso_solver(X, y, w, regs[i])))
    return W

def plot_regpath(df, W, regs, features):
    ids = [df.columns.get_loc(feature) - 1 for feature in features]

    plt.title('Regularization Paths')
    plt.xlabel('log(lambda)')
    plt.ylabel('weights')
    plt.plot(np.log(regs), W[:, ids], marker='o')
    plt.legend(features, fontsize=8)
    plt.show()

def plot_sqerr(X, y, W, regs):
    err = ((np.matmul(X, W.T).T - y) ** 2).sum(axis=1)
    plt.title('squared erros:')
    plt.xlabel('log(lambda)')
    plt.xlabel('squared errors')
    plt.plot(np.log(regs), err, marker='o')
    plt.show()

def plot_nonzero(W, regs):
    W_count = ((W != 0) * 1).sum(axis=1)
    plt.title('number of non-zero coeffcients')
    plt.xlabel('lambda')
    plt.ylabel('# of non-zero coeff.')
    plt.plot(regs, W_count, marker='o')
    plt.show()

def main():
    X_train, y_train, X_test, y_test, df_train, df_test = load_data('data')
    regs = np.array([600.0 / (2 ** i) for i in range(10)])
    W = lasso_models(X_train, y_train, regs)

if __name__ == '__main__':
    main()