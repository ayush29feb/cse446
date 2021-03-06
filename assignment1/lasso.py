import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(datadir, validation=False, split=0.9, rand_seed=200):
    df_train = pd.read_table(datadir + '/crime-train.txt')
    df_test = pd.read_table(datadir + '/crime-test.txt')

    
    X_test = df_test.iloc[:, 1:].values
    y_test = df_test.iloc[:, :1].values.reshape(-1)

    if validation:
        df_train_split = df_train.sample(frac=split, random_state=rand_seed)
        df_val_split = df_train.drop(df_train_split.index)
        
        X_train = df_train_split.iloc[:, 1:].values
        y_train = df_train_split.iloc[:, :1].values.reshape(-1)
        X_val = df_val_split.iloc[:, 1:].values
        y_val = df_val_split.iloc[:, :1].values.reshape(-1)
        
        return (X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_test)
    else:
        X_train = df_train.iloc[:, 1:].values
        y_train = df_train.iloc[:, :1].values.reshape(-1)
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

def plot_regpath(W, regs, features, ids):
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
    plt.ylabel('squared errors')
    plt.plot(np.log(regs), err, marker='o')
    plt.show()

def plot_nonzero(W, regs):
    W_count = ((W != 0) * 1).sum(axis=1)
    plt.title('number of non-zero coeffcients')
    plt.xlabel('lambda')
    plt.ylabel('# of non-zero coeff.')
    plt.plot(regs, W_count, marker='o')
    plt.show()

def tune_reg(X, y, W, regs):
    err = ((np.matmul(X, W.T).T - y) ** 2).sum(axis=1)
    best_reg = regs[np.argmin(err)]
    return best_reg

def max_min_w(df, W, regs, best_reg):
    best_w = W[np.where(regs==best_reg), :]
    max_w_i = np.argmax(best_w)
    min_w_i = np.argmin(best_w)
    return df.columns[[max_w_i + 1, min_w_i + 1]].tolist()

def main():
    X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_test = load_data('data', validation=True, split=0.9)
    regs = np.array([600.0 / (2 ** i) for i in range(10)])
    W = lasso_models(X_train, y_train, regs)

    features = ['agePct12t29', 'pctWSocSec', 'PctKids2Par', 'PctIlleg', 'HousVacant']
    ids = [df_train.columns.get_loc(feature) - 1 for feature in features]
    plot_regpath(W, regs, features, ids)

    plot_sqerr(X_train, y_train, W, regs)
    plot_sqerr(X_test, y_test, W, regs)
    plot_nonzero(W, regs)

    plot_sqerr(X_val, y_val, W, regs)
    best_reg = tune_reg(X_val, y_val, W, regs)
    print 'best validation error at reg=' + str(best_reg)

    max_w_i, min_w_i = max_min_w(df_train, W, regs, best_reg)
    print 'Maximum Parameter: ' + max_w_i + ' and Minimum Parameter: ' + min_w_i

if __name__ == '__main__':
    main()