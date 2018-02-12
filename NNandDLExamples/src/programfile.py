import mnist_loader
# import network #exercise 1, 2, 3
# import network2 #exercise 4-10
import alynetwork2

# print(mnist_loader)
# TODO: magictypes JULIAN AVI ALLISON
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# print(training_data[0])

# # Exercise 1 -- < 5 minutes
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# Exercise 2 -- 8 minutes
# net = network.Network([784, 10])
# net.SGD(training_data, 100, 10, 0.5, test_data=test_data)

# Exercise 3 --
# net = network.Network([784, 10])
# net.SGD(training_data, 600, 10, 0.50, test_data=test_data)
# Epoch 599: 7438 / 10000 .. terrible, obviously.

# Exercise 4 -- Chapter 3
# Cross-entropy to classify MNIST digits
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

# # Exercise 5 -- Chapter 3
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)
# # Epoch 399 training complete
# # Cost on training data: 0.00359133710628
# # Accuracy on evaluation data: 8191 / 10000

# # Exercise 6 -- Chapter 3
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, lmbda = 0.1, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
# # Epoch 399 training complete
# # Cost on training data: 0.11593782793
# # Accuracy on training data: 1000 / 1000
# # Cost on evaluation data: 0.846376388988
# # Accuracy on evaluation data: 8786 / 10000

# # Exercise 7 -- Chapter 3
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, lmbda = 5.0, monitor_evaluation_accuracy=True, monitor_training_accuracy=True)
# # Epoch 29 training complete
# # Accuracy on training data: 48525 / 50000
# # Accuracy on evaluation data: 9622 / 10000

# # Exercise 8 -- Chapter 3
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.5, evaluation_data=validation_data, lmbda = 5.0, monitor_evaluation_accuracy=True)
# # Epoch 29 training complete
# # Accuracy on evaluation data: 9776 / 10000

# # Exercise 9 -- Chapter 3 (Gaussian random normalized mean 0, standard deviation of 1)
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.1, evaluation_data=validation_data, lmbda = 5.0, monitor_evaluation_accuracy=True)
# # Epoch 29 training complete
# # Accuracy on evaluation data: 9608 / 10000

# # Exercise 10 -- Chapter 3  (Gaussian random normalized mean 0, standard deviation of 1/((n(in))^(1/2))...
# # ... one over the square root of the number of number of input weights.
# # note that we dropped "large_weight_initializer()" to do that.
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.SGD(training_data, 30, 10, 0.1, evaluation_data=validation_data, lmbda = 5.0, monitor_evaluation_accuracy=True)
# # Epoch 19 training complete
# # Accuracy on evaluation data: 9610 / 10000
# # Epoch 25 training complete
# # Accuracy on evaluation data: 9640 / 10000
# # Epoch 29 training complete
# # Accuracy on evaluation data: 9618 / 10000

# Exercise 11 -- Chapter 3 -- now using alynetwork2
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = alynetwork2.Network([784, 30, 10], cost=alynetwork2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)
