import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, :1].values.reshape(-1)
    features = df.iloc[:, 1:].columns.values

    return (X, y, features)

def linear_loss(X, y , w):
    y_ = np.matmul(X, w) # X (N, D) w(D,) = Xw (N,)
    if y is None:
        return (y_ > 0.5) * 1
    
    loss = np.sum((y - (y_ > 0.5) * 1) ** 2)
    dw = -2 * np.matmul(X.T, y - y_)
    return (loss, dw)

def logistic_loss(X, y, w):
    y_ = (1 + np.exp(-1 * np.matmul(X, w))) ** -1 # X (N, D) w(D,) = Xw (N,)
    if y is None:
        return (y_ > 0.5) * 1.0

    loss = np.sum((y - (y_ > 0.5) * 1) ** 2)
    dw = -1 * np.matmul(X.T, y - y_)
    return (loss, dw)

def train(X, y, w, loss_fn, eta, itr, logs=None):
    N = X['train'].shape[0]

    if logs is not None:
        logs['total_loss'] = 0
        logs['train_losses'] = {'x': None, 'y': []}
        logs['test_losses'] = {'x': None, 'y': []}
        logs['l2s'] = {'x': None, 'y': []}

    W = np.zeros(w.shape)

    for i in range(itr):
        loss, dw = loss_fn(X['train'][i % N, :].reshape(1, -1), y['train'][i % N].reshape(1,), w)
        w -= eta * dw
        
        if i >= (itr - N):
            W += w

        if logs is not None:
            logs['total_loss'] += loss
            if (i + 1) % 100 == 0:
                # train loss - (4.2.b.i)
                logs['train_losses']['y'].append(1.0 * logs['total_loss'] / (i + 1))
                # test loss - (4.2.b.iii)
                test_loss, _ = loss_fn(X['test'], y['test'], w)
                logs['test_losses']['y'].append(test_loss)
            if (i + 1) % N == 0:
                # l2 of weights - (4.2.b.ii)
                logs['l2s']['y'].append(np.sum(w ** 2))
    
    if logs is not None:
        logs['train_losses']['x'] = (np.arange(len(logs['train_losses']['y'])) + 1) * 100
        logs['test_losses']['x'] = (np.arange(len(logs['test_losses']['y'])) + 1) * 100
        logs['l2s']['x'] = (np.arange(len(logs['l2s']['y'])) + 1) * N
        del logs['total_loss']
    
    return W / N

def plot_graphs(logistic_logs, linear_logs, etas):
    for eta in etas:
        plt.plot(logistic_logs[eta]['train_losses']['x'], logistic_logs[eta]['train_losses']['y'])
        plt.title('Logistic Reg. Training Avg. Loss: eta=' + str(eta))
        plt.xlabel('time step t')
        plt.ylabel('loss')
        plt.show()
    
    for eta in etas:
        plt.plot(logistic_logs[eta]['l2s']['x'], logistic_logs[eta]['l2s']['y'])
        plt.title('Logistic Reg. L2(Weight): eta=' + str(eta))
        plt.xlabel('time step t')
        plt.ylabel('L2(weights)')
        plt.show()
    
    for eta in etas:
        plt.plot(logistic_logs[eta]['test_losses']['x'], logistic_logs[eta]['test_losses']['y'])
        plt.title('Logistic Reg. Test Losses: eta=' + str(eta))
        plt.xlabel('time step t')
        plt.ylabel('loss')
        plt.show()
    
    for eta in etas:
        plt.plot(linear_logs[eta]['train_losses']['x'], linear_logs[eta]['train_losses']['y'])
        plt.title('Linear Reg. Training Avg. Loss: eta=' + str(eta))
        plt.xlabel('time step t')
        plt.ylabel('loss')
        plt.show()
    
    for eta in etas:
        plt.plot(linear_logs[eta]['l2s']['x'], linear_logs[eta]['l2s']['y'], marker='o')
        plt.title('Linear Reg. L2(Weight): eta=' + str(eta))
        plt.xlabel('time step t')
        plt.ylabel('L2(weights)')
        plt.show()

    for eta in etas:
        plt.plot(linear_logs[eta]['test_losses']['x'], linear_logs[eta]['test_losses']['y'])
        plt.title('Linear Reg. Test Losses: eta=' + str(eta))
        plt.xlabel('time step t')
        plt.ylabel('loss')
        plt.show()

def main():
    X_train, y_train, features = load_data('dataset/train.csv')
    X_test, y_test, _ = load_data('dataset/test.csv')

    X_train = np.append(X_train, np.ones((X_train.shape[0], 1)), axis=1)
    X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)

    X = {'train': X_train, 'test': X_test}
    y = {'train': y_train, 'test': y_test}

    N, D = X_train.shape
    etas = [0.8, 1e-3, 1e-5]

    # plots
    logistic_logs = {}
    linear_logs = {}
    for eta in etas:
        logistic_logs[eta] = {}
        train(X, y, np.zeros(D), logistic_loss, eta, 10 * N, logistic_logs[eta])
        linear_logs[eta] = {}
        train(X, y, np.zeros(D), linear_loss, eta, 10 * N, linear_logs[eta])
    
    plot_graphs(logistic_logs, linear_logs, etas)

    # Best Model
    logistic_w = {}
    logistic_logs = {}
    for eta in etas:
        logistic_logs[eta] = {}
        logistic_w[eta] = train(X, y, np.zeros(D), logistic_loss, eta, 100000, logs=logistic_logs[eta])

    best_eta = etas[0]
    best_loss, _ = logistic_loss(X['train'], y['train'], logistic_w[best_eta])
    for eta in etas[1:]:
        loss, _ = logistic_loss(X['train'], y['train'], logistic_w[eta])
        if loss < best_loss:
            best_eta = eta
            best_loss = loss

    print 'best eta=' + str(best_eta) + ' with loss=' + str(best_loss)
    features_ = np.array(['BMI', 'insulin', 'PGC'])
    ids = [np.argmax(features == f) for f in features_]
    print str(features_) + " = " + str(logistic_w[best_eta][ids])

if __name__ == '__main__':
    main()
