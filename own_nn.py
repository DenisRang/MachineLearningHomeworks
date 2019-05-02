"""
HOMEWORK:

Part 2: make a new NN with a topology 2-5-5-1 by adding one more hidden layer
It means that you need to recalculate blocks:
 - initialization parameters
 - predict
 - updating parameters

"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


np.random.seed(0)
#download the dataset
X, labels = sklearn.datasets.make_moons(200, noise=0.20)
y = labels.reshape(-1, 1)

#topology of our Neural Network: 2-5-5-1
# 2: input layer
# 5: layer 1
# 5: layer 2
# 1: layer 3
input_size = 2
layer1 = 5
layer2 = 5
layer3 = 1

#initialize parameters
W1 = 2*np.random.random((input_size, layer1)) - 1   #weights matrix of the first layer
b1 = 2*np.random.random((1,          layer1)) - 1   #bias vector of the first layer
W2 = 2*np.random.random((layer1, layer2)) - 1   #weights matrix of the second layer
b2 = 2*np.random.random((1,      layer2)) - 1   #bias vector of the second layer
W3 = 2*np.random.random((layer2, layer3)) - 1   #weights matrix of the third layer
b3 = 2*np.random.random((1,      layer3)) - 1   #bias vector of the third layer


#lr: learning rate for parameters updating
lr = 0.5
for j in range(1000):
		# FEED FORWARD PROPAGATION
    z1 = np.dot(X,W1)  + b1   #first layer perceptrons
    a1 = sigmoid(z1)          #first layer activation
    
    z2 = np.dot(a1,W2) + b2   #second layer perceptrons
    a2 = sigmoid(z2)          #second layer activation
    
    z3 = np.dot(a2,W3) + b3   #second layer perceptrons
    a3 = sigmoid(z3)          #second layer activation

    
    # BACKWARD PROPAGATION
    # C = 1/2 * (a3 - y)^2   <-- loss function
    # ∂C/∂a3 = a3 - y        <-- simple calculus exercise
    # partial derivative of sigmoid = a2*(1-a2) 
    a3_delta = (a3 - y) * (a3 * (1-a3))
    a2_delta = a3_delta.dot(W3.T) * (a2 * (1-a2))
    a1_delta = a2_delta.dot(W2.T) * (a1 * (1-a1))

    # UPDATE PARAMETERS
    W1 = W1 - lr * X.T.dot(a1_delta)
    b1 = b1 - lr * a1_delta.sum(axis=0, keepdims=True)

    W2 = W2 - lr * a1.T.dot(a2_delta)
    b2 = b2 - lr * a2_delta.sum(axis=0, keepdims=True)

    W3 = W3 - lr * a2.T.dot(a3_delta)
    b3 = b3 - lr * a3_delta.sum(axis=0, keepdims=True)


def predict(X):
	"""
	feedforward
	input: X
	output: final layer output
	W, b: parameters of the layer
	z: perceptron = ∑w*x + b
	a: activation of perceptron = sigmoid(z)
	"""
	z1 = np.dot(X,W1)  + b1   
	a1 = sigmoid(z1)          
	z2 = np.dot(a1,W2) + b2   
	a2 = sigmoid(z2)          
	z3 = np.dot(a2,W3) + b3   
	a3 = sigmoid(z3)

	return a3


_, axes = plt.subplots(1, 2, figsize=(11,5), sharey=True)

axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Spectral)
axes[0].set_xlim((-1.5, 2.5))
axes[0].set_ylim((-1, 1.5))

test_x1 = np.linspace(-1.5, 2.5, 20)
test_x2 = np.linspace(-1,   1.5, 20)
for x1 in test_x1:
    for x2 in test_x2:
        y = predict([[x1, x2]])
        color = 'blue' if y > 0.5 else 'red'
        axes[1].scatter(x1, x2, c=color)
plt.show()

