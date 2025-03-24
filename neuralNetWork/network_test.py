import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.stochastic_gradient_descent(training_data, 30, 10, 0.75, test_data=test_data)