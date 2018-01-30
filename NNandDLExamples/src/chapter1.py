import mnist_loader
import NetworkAly

# print(mnist_loader)
# TODO: magictypes JULIAN AVI ALLISON
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# print(training_data[0])

# # def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):

# # SGD 30 epochs, mini-batch of 10, learing rate 3.0
# net = NetworkAly.NetworkAly([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# # SGD 100 epochs, minibatch 10, learning rate 0.001, 100 hidden nodes (one layer)
# net = NetworkAly.NetworkAly([784, 100, 10])
# net.SGD(training_data, 30, 10, 0.001, test_data=test_data)

# net = NetworkAly.NetworkAly([784, 30, 10])
# net.SGD(training_data, 30, 10, 100, test_data=test_data)

# try it wiht only 2 layers
net = NetworkAly.NetworkAly([784, 10])
net.SGD(training_data, 600, 10, .2, test_data=test_data)