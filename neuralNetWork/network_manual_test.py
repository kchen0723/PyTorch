import network
import numpy as np

# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
net = network.Network([2, 2, 2])
net.biases = np.array([[0.35, 0.35], [0.6, 0.6]])
net.weights = np.array([[[0.15, 0.20], [0.25, 0.30]], [[0.40, 0.45], [0.50, 0.55]]])
training_data = np.array([[[0.05, 0.10], [0.01, 0.99]]])
net.stochastic_gradient_descent(training_data, 30, 10, 0.75)
predicate = net.feedforward(training_data[0][0])
print(predicate)

# biases = np.array([[0.35, 0.35], [0.6, 0.6]])
# weights = np.array([[[0.15, 0.20], [0.25, 0.30]], [[0.40, 0.45], [0.50, 0.55]]])
# input = np.array([0.05, 0.10])

# activation = input
# activations = []
# for b, w in zip(biases, weights):
#     npdot = np.dot(w, activation)
#     pre_activation = npdot + b
#     activation = 1.0/(1.0 + np.exp(-pre_activation))
#     activations.append(activation)

# print(activations)