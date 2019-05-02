import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def split_dataframe(x, y, fraction):
    _snuffle_dataframe(x, y)
    fraction_i = int(size * fraction)
    x_train = x.iloc[:fraction_i, :]
    y_train = y[:fraction_i]
    x_test = x.iloc[fraction_i:, :]
    y_test = y[fraction_i:]
    return x_train, x_test, y_train, y_test


def _snuffle_dataframe(x, y):
    for i in range(size):
        i_swap = np.random.randint(0, size - 1)
        x.iloc[i], x.iloc[i_swap] = x.iloc[i_swap], x.iloc[i]
        y.iloc[i], y.iloc[i_swap] = y.iloc[i_swap], y.iloc[i]


def generate_data(b1, b0, size, x_range=(-10, 10), noise_mean=0,
                  noise_std=1):

    noise = np.random.normal(noise_mean, noise_std, size)
    rnd_vals = np.random.rand(size)
    data_x = x_range[1] * rnd_vals + x_range[0] * (1 - rnd_vals)
    data_y = b1 * data_x + b0 + noise

    return data_x, data_y


def gradient_descent(x, y):
    """
    input:
    x, y - data features

    output:
    b1, b0 - predicted parameters of data
    """
    b1 = np.random.uniform()
    b0 = np.random.uniform()
    epsilon = 0.001
    N = len(y)
    alpha = 0.001
    curError = 0
    prevError = 1000
    curError = np.mean((y - (b0 + b1 * x)) ** 2)
    while abs(prevError - curError) > epsilon:
        dJ1 = - np.mean( 2 * x * (y - (b0 + b1 * x)))
        dJ0 = - np.mean( 2 * (y - (b0 + b1 * x)))
        b1 -= alpha * dJ1
        b0 -= alpha * dJ0
        prevError = curError
        curError = np.mean((y - (b0 + b1 * x)) ** 2)
    mse = np.mean((y - (b0 + b1 * x)) ** 2)
    return b1, b0, mse


def least_squares(x, y):
    """
    input:
    x, y - data features

    output:
    b1, b0 - predicted parameters of data
    """
    mean_x = x.mean()
    mean_y = y.mean()

    b1 = np.dot(y - mean_y, x - mean_x) / np.dot(x - mean_x, x - mean_x)
    b0 = mean_y - b1 * mean_x
    mse = np.mean((y - (b0 + b1 * x)) ** 2)

    return b1, b0, mse


def animate(data_x, data_y, true_b1, true_b0, b1, b0, x_range=(-10, 10),
            label="Least squares"):
    plt.scatter(data_x, data_y)
    plt.plot([x_range[0], x_range[1]],
             [x_range[0] * true_b1 + true_b0, x_range[1] * true_b1 + true_b0],
             c="r", linewidth=2, label="True")
    plt.plot([x_range[0], x_range[1]],
             [x_range[0] * b1 + b0, x_range[1] * b1 + b0],
             c="g", linewidth=2, label=label)

    plt.legend()
    plt.show()


### Parameters for data generation ###
true_b1 = 2.5
true_b0 = -7
size = 100
x_range = (0, 10)
noise_mean = 0
noise_std = 1

# Generate the data
data_x, data_y = generate_data(true_b1, true_b0, size,
                               x_range=x_range,
                               noise_mean=noise_mean,
                               noise_std=noise_std)

# Visualize the data
print("true b1 : {}\ntrue b0 : {}".format(true_b1, true_b0))
# Predict data's parameters
b1, b0, mse = least_squares(data_x, data_y)
print("calculated b1 : {}\ncalculated b0 : {}\nMSE : {}\n".format(b1, b0, mse))
b1, b0, mse = gradient_descent(data_x, data_y)
print("calculated b1 : {}\ncalculated b0 : {}\nMSE : {}\n".format(b1, b0, mse))
animate(data_x, data_y, true_b1, true_b0, b1, b0, x_range=x_range)
