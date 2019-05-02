import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def gradient_decent(X, y):
    N = len(y)
    epsilon = 0.01
    alfa = 0.001
    b0 = np.random.uniform()
    b1 = np.random.uniform()
    prev_error = 1000;
    cur_error = cost_function(X, y, b0, b1)

    while (np.abs(cur_error - prev_error) > epsilon):
        dL0 = 0
        dL1 = 0
        for i in range(N):
            dL0 += hat_probability(X[i], b0, b1) - y[i]
            dL1 += X[i] * (hat_probability(X[i], b0, b1) - y[i])
        b0 -= alfa * dL0 / N
        b1 -= alfa * dL0 / N
        prev_error = cur_error
        cur_error = cost_function(X, y, b0, b1)
    return b0, b1


def cost_function(X, y, b0, b1):
    N = len(y)
    cost_class1 = 0
    cost_class2 = 0
    for i in range(N):
        cost_class1 += y[i] * np.log(hat_probability(X[i], b0, b1))
        cost_class1 += (1 - y[i]) * np.log(1 - hat_probability(X[i], b0, b1))
    return -(cost_class1 + cost_class2) / N


def predict(X, b0, b1, threshold):
    N = len(X)
    y_pred = []

    for i in range(N):
        if (hat_probability(X[i], b0, b1) > threshold):
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred


def hat_probability(x, b0, b1):
    return (np.e ** (b0 + b1 * x)) / (1 + np.e ** (b0 + b1 * x))


def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))


iris = load_iris()
X = iris.data
y = iris.target
N = len(y)

# take 1 feature and 2 of 3 classes of response
X = X.take(indices=0, axis=1)
indices_to_extract = []
for i in range(N):
    if (y[i] != 2):
        indices_to_extract.append(i)
X = X.take(indices_to_extract)
X = X.reshape((len(X), 1))
y = y.take(indices_to_extract)

# split on train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

threshold = 0.9
for i in range(100):
    threshold += 0.001
    b0, b1 = gradient_decent(X_train, y_train)
    y_pred = predict(X_test, b0, b1, threshold)
    acc = accuracy(y_pred, y_test)
    print(acc)
