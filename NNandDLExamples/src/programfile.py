import mnist_loader
import network

# print(mnist_loader)
# TODO: magictypes JULIAN AVI ALLISON
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# print(training_data[0])

# # Exercise 1 -- < 5 minutes
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# Exercise 2 -- 8 minutes
# net = network.Network([784, 10])
# net.SGD(training_data, 100, 10, 0.5, test_data=test_data)

# Exercise 3 --
net = network.Network([784, 10])
net.SGD(training_data, 600, 10, 0.50, test_data=test_data)
