import random
import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1 :]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[: -1], sizes[1 :])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        if test_data:
            n_test = len(test_data)
            n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch{0} completed".format(j))

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nbla_w = self.back_propagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nb + dnw for nb, dnw in zip(nabla_w, delta_nbla_w)]
            self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def back_propagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        pre_activations = []
        # forward pass
        for b, w in zip(self.biases, self.weights):
            pre_activation = np.dot(w, activation) + b
            pre_activations.append(pre_activation)
            activation = sigmoid(pre_activation)
            activations.append(activation)
            
        # backward pass
        # https://www.youtube.com/watch?v=tIeHLnjs5U8
        # a = f(w . x + b), so c = y - a = y - f(w . x + b)
        # so delta(b) = cost * sigmoid_prime, delta(w) = delta(b) * activation[previos layer]
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(pre_activations[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for i in range(2, self.num_layers):
            pre_activation = pre_activations[-i]
            local_gradient = sigmoid_prime(pre_activation)
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * local_gradient
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i - 1].transpose())
        return(nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return(output_activations - y)

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * ( 1 - sigmoid(z))