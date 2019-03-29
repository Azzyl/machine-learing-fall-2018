"""
HOMEWORK:
this is an example of NN with a topology "2-5-1" and sigmoid activation on each layer

Part 1: write 2 more activation functions: tanh and relu.
	a)Implement new NN with the same topology "2-5-1" and relu activation on the first layer
	b)Implement new NN with the same topology "2-5-1" and tanh activation on the first layer

Part 2: make a new NN with a topology 2-5-5-1 by adding one more hidden layer
It means that you need to recalculate blocks with initialization parameters, predict and updating parameters

"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivation(a):
    return a * (1 - a)

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def tanh_derivation(a):
    return 1 - np.power(a, 2)

def relu(x):
    return np.maximum(0, x)

def relu_derivation(a):
    return 1 * (a > 0)

def predict(X, layers, weights, biases, functions_):
    """
    feedforward
    input: X
    output: final layer output
    W1, b1: parameters of the first  layer
    W2, b2: parameters of the second layer
    z: perceptron = ∑w*x + b
    a: activation of perceptron = sigmoid(z)
    """
    a = []
    for i in range(len(layers) - 1):
        if i == 0:
            a.append(functions_[i](np.dot(X, weights[i]) + biases[i]))
        else:
            a.append(functions_[i](np.dot(a[i - 1], weights[i]) + biases[i]))

    return a[len(layers) - 2]

functions = {
    'sigm': sigmoid,
    'tanh': tanh,
    'relu': relu
}

derivations = {
    'sigm': sigmoid_derivation,
    'tanh': tanh_derivation,
    'relu': relu_derivation
}

np.random.seed(0)
# download the dataset
X, labels = sklearn.datasets.make_moons(200, noise=0.20)
y = labels.reshape(-1, 1)

def fit_net(X, layers, _functions, _derivations, y=None):
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        weights.append(2 * np.random.random((layers[i], layers[i + 1])) - 1)
        biases.append(2 * np.random.random((1, layers[i + 1])) - 1)

    lr = 0.1
    for j in range(20000):
        activations = []
        deltas = []

        # FEED FORWARD PROPAGATION
        activations.append(_functions[0](np.dot(X, weights[0]) + biases[0]))
        for i in range(1, len(layers) - 1):
            activations.append(_functions[i](np.dot(activations[i - 1], weights[i]) + biases[i]))

        # BACKWARD PROPAGATION
        for i in range(len(layers) - 2, -1, -1):
            if i == len(layers) - 2:
                deltas.append((activations[i] - y) * derivations_[i](activations[i]))
            else:
                deltas.append(deltas[i-1] @ weights[i+1].T * derivations_[i](activations[i]))
        deltas =  list(reversed(deltas))

        # UPDATE PARAMETERS
        for i in range(len(layers) - 1):
            if i == 0:
                weights[i] -= lr * (X.T @ deltas[i])
            else:
                weights[i] -= lr * (activations[i - 1].T @ deltas[i])
            biases[i] -= lr * deltas[i].sum(axis=0, keepdims=True)
    return weights, biases

def show_plot(X, layers, weights, biases, functions_):
    _, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Spectral)
    axes[0].set_xlim((-1.5, 2.5))
    axes[0].set_ylim((-1, 1.5))

    test_x1 = np.linspace(-1.5, 2.5, 20)
    test_x2 = np.linspace(-1, 1.5, 20)
    for x1 in test_x1:
        for x2 in test_x2:
            y = predict([[x1, x2]], layers, weights, biases, functions_)
            color = 'blue' if y > 0.5 else 'red'
            axes[1].scatter(x1, x2, c=color)
    plt.show()

# topology of our Neural Network: 2-5-1
# 2: input layer
# 5: layer 1
# 1: layer 2

# First case
functions_ = [functions['sigm']] * 3
derivations_ = [derivations['sigm']] * 3
w, b = fit_net(X, [2, 5, 1], functions_, derivations_, y)
show_plot(X, [2, 5, 1], w, b, functions_)

# Seconds case
functions_ = [functions['tanh'], functions['sigm'], functions['sigm']]
derivations_ = [derivations['tanh'], derivations['sigm'], derivations['sigm']]
w, b = fit_net(X, [2, 5, 1], functions_, derivations_, y)
show_plot(X, [2, 5, 1], w, b, functions_)

# Third case
functions_ = [functions['relu'], functions['sigm'], functions['sigm']]
derivations_ = [derivations['relu'], derivations['sigm'], derivations['sigm']]
w, b = fit_net(X, [2, 5, 1], functions_, derivations_, y)
show_plot(X, [2, 5, 1], w, b, functions_)

# Fourth case
functions_ = [functions['sigm'], functions['sigm'], functions['sigm'], functions['sigm']]
derivations_ = [derivations['sigm'], derivations['sigm'], derivations['sigm'], derivations['sigm']]
w, b = fit_net(X, [2, 5, 5, 1], functions_, derivations_, y)
show_plot(X, [2, 5, 5, 1], w, b, functions_)