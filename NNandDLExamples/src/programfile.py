import mnist_loader
print(mnist_loader)
# TODO: magictypes JULIAN AVI ALLISON
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print(training_data[0])