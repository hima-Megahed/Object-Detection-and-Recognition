import numpy as np
import math
import matplotlib.pyplot as plt
from DataManipulation import TrainingData


class BackPropagation:

    def __init__(self):
        # Initializing weight vector with random values
        self.Network = None
        self.__best_Network = None
        self.__best_initialize_Network = None

    def main_algorithm(self, features, eta, epochs, bias, threshold,
                       stopping_criteria, activation_function,
                       num_hidden_layer, num_neurons_layer, n_samples):
        """This function will run the back propagation algorithm"""

        print("The Network will train using: BackPropagation Classifier")

        # Initialize Neural Network
        self.Network = self.initialize_network(num_neurons_layer,
                                num_hidden_layer)
        # Number of epochs
        if stopping_criteria == 1:
            print("Stopping condition: Number of Epochs")

            minsum = 50000000.0
            # loop through number of epochs
            for i in range(epochs):
                sample_ind = 0
                sum_error = 0
                # loop through number of samples
                for sample in features:
                    outputs = self.forward_propagate(self.Network, sample,
                                                     bias, activation_function)

                    expected = self.__get_expected(sample_ind)
                    sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in
                                      range(len(expected))])
                    self.backward_propagate_error(self.Network,
                                                  expected,
                                                  activation_function)
                    self.update_weights(self.Network, sample, eta, bias)
                    sample_ind += 1
                if minsum > sum_error:
                    minsum = sum_error
                    self.__best_Network = self.Network
                print('> epoch=%d, error=%.5f, Minimum is: %.5f'% (i, sum_error, minsum))
        # Threshold MSE
        elif stopping_criteria == 2:
            print("Stopping condition: Mean Square Error")
            epoch_ind = 1
            mse = 10000000.0
            mse_errors = [0] * 5
            min_mse = 100000

            while epoch_ind < 2000:
                sample_ind = 0
                # loop through number of samples
                for sample in features:
                    self.forward_propagate(self.Network, sample,
                                           bias, activation_function)

                    expected = self.__get_expected(sample_ind)
                    self.backward_propagate_error(self.Network,
                                                  expected,
                                                  activation_function)
                    self.update_weights(self.Network, sample, eta, bias)
                    sample_ind += 1

                mse, mse_errors = self.compute_mean_square_error(
                    self.Network[-1:][0], mse_errors, n_samples)
                if min_mse > mse:
                    min_mse = mse
                    self.__best_Network = self.Network
                print("> epoch:{}, Mean Square Error:{}, Minimum MSE:{}".format(epoch_ind, mse, min_mse))
                epoch_ind += 1
                #if mse <= threshold:
                #    break

        # Cross Validation
        else:
            print("Stopping condition: Cross Validation")
            max_acc = 0
            # loop through number of epochs
            for i in range(epochs):
                model_output = actual_output = []
                # loop through number of samples
                for sample_ind in range(20):
                    sample = features[sample_ind]
                    self.forward_propagate(self.Network, sample,
                                           bias, activation_function)

                    expected = self.__get_expected(sample_ind)
                    self.backward_propagate_error(self.Network,
                                                  expected,
                                                  activation_function)
                    self.update_weights(self.Network, sample, eta, bias)

                # Validate after 50 epochs
                if i % 50 == 0 and i != 0:
                    model_output, actual_output = self.cross_validate(
                        features, bias, activation_function, num_hidden_layer,
                        num_neurons_layer)
                    # Computing Confusion Matrix
                    confusion_matrix = [[0 for x in range(5)]
                                        for y in range(5)]
                    for s in range(len(model_output)):
                        y = actual_output[s]
                        confusion_matrix[y - 1][model_output[s] - 1] += 1
                    print("Model Output: ", model_output)
                    print("Actual Output: ", actual_output)
                    # Calculating Overall accuracy
                    # 5 number of tests
                    acc = (np.sum([confusion_matrix[i][i]
                                  for i in range(5)]) / 5) * 100
                    if max_acc < acc:
                        max_acc = acc
                        self.__best_Network = self.Network
                    print("Accuracy is: {acc:.3f}, Best Accuracy: {max_acc:.3f}")

                    asdf = 0

    @staticmethod
    def compute_mean_square_error(error, mse_errors, n_samples):
        err_sum = 0.0
        tmp = []
        for i in range(len(error)):
            mse_errors[i] += error[i]['delta']**2

        for i in range(len(mse_errors)):
            tmp.append(mse_errors[i] / (2 * n_samples))
        err_sum = np.sum(tmp) / 5

        return err_sum, mse_errors

    @staticmethod
    def sigmoid(net):
        return 1.0 / (1.0 + np.exp(-net))

    @staticmethod
    def hyperbolic_tangent(x):
        return np.tanh(x)

    def derivative_transfer(self, activation_function, x):
        if activation_function == 1:
            return self.sigmoid_derivative(x)
        else:
            return self.hyperbolic_tangent_derivative(x)

    @staticmethod
    def sigmoid_derivative(net):
        return net * (1.0 - net)

    @staticmethod
    def hyperbolic_tangent_derivative(x):
        return 1- np.power(np.tanh(x),2)

    def cross_validate(self, features, bias, activation_function,
                       num_hidden_layer,num_neurons_layer):
        model_output = []
        actual_output = []
        for sample_ind in range(20, 25):
            # getting input vector
            sample = features[sample_ind]
            outputs = self.forward_propagate(self.Network, sample, bias,
                                             activation_function)
            f = outputs.index(max(outputs))
            model_output.append(outputs.index(max(outputs)) + 1)
            actual_output.append(self.get_sample_ind_(sample_ind))
        return model_output, actual_output

    @staticmethod
    def __get_expected(num):
        if num in [0, 5, 10, 15, 20]:
            return [1, 0, 0, 0, 0]
        elif num in [1, 6, 11, 16, 21]:
            return [0, 1, 0, 0, 0]
        elif num in [2, 7, 12, 17, 22]:
            return [0, 0, 1, 0, 0]
        elif num in [3, 8, 13, 18, 23]:
            return [0, 0, 0, 1, 0]
        elif num in [4, 9, 14, 19, 24]:
            return [0, 0, 0, 0, 1]

    @staticmethod
    def shuffle_data(data):
        new_data = [
            data[0],
            data[5],
            data[10],
            data[15],
            data[20],
            data[1],
            data[6],
            data[11],
            data[16],
            data[21],
            data[2],
            data[7],
            data[12],
            data[17],
            data[22],
            data[3],
            data[8],
            data[13],
            data[18],
            data[23],
            data[4],
            data[9],
            data[14],
            data[19],
            data[24],
        ]
        return new_data

    @staticmethod
    def get_sample_ind_(index):
        if index in [0, 5, 10, 15, 20]:
            return 1
        elif index in [1, 6, 11, 16, 21]:
            return 2
        elif index in [2, 7, 12, 17, 22]:
            return 3
        elif index in [3, 8, 13, 18, 23]:
            return 4
        elif index in [4, 9, 14, 19, 24]:
            return 5

    #########################################################################
    @staticmethod
    def initialize_network(n_hidden_neurons, n_hidden_layers):
        network = list()
        first_hidden_layer = [{'weights': [np.random.rand(1)[0]
                                           for i in range(24 + 1)]}
                              for i in range(n_hidden_neurons)]
        network.append(first_hidden_layer)
        for j in range(n_hidden_layers - 1):
            hidden_layer = [{'weights': [np.random.rand(1)[0]
                                         for i in range(n_hidden_neurons + 1)]}
                            for i in range(n_hidden_neurons)]
            network.append(hidden_layer)
        output_layer = [
            {'weights': [np.random.rand(1)[0]
                         for i in range(n_hidden_neurons + 1)]}
            for i in range(5)]
        network.append(output_layer)
        return network

    # Calculate neuron activation for an input
    @staticmethod
    def activate(weights, inputs, bias):
        activation = weights[0] * bias
        for i in range(1, len(weights)):
            activation += weights[i] * inputs[i - 1]
        #for i in range(1, len(weights) - 1):
        #   activation += weights[i] * inputs[i]
        return activation

    # Transfer neuron activation
    def transfer(self, net, activation_method):
        return self.sigmoid(net) if activation_method == 1 \
            else self.hyperbolic_tangent(net)

    # Forward propagate input to a network output
    def forward_propagate(self, network, row_data, bias, activation_method):
        inputs = row_data
        for layer in network:
            new_inputs = []
            for neuron in layer:
                net = self.activate(neuron['weights'], inputs, bias)
                neuron['output'] = self.transfer(net, activation_method)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # Calculate the derivative of an neuron output
    def transfer_derivative(self, output, activation_method):
        return self.sigmoid_derivative(output) if activation_method == 1 \
            else self.hyperbolic_tangent_derivative(output)

    # Back propagate error and store in neurons
    def backward_propagate_error(self, network, expected, activation_method):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i == len(network) - 1:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            else:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(
                    neuron['output'], activation_method)

    @staticmethod
    def update_weights(network, data_row, l_rate, bias):
        for i in range(len(network)):
            inputs = data_row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(1, len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] \
                                            * inputs[j-1]
                neuron['weights'][0] += l_rate * neuron['delta'] * bias

    def test(self, samples, bias, activation_function):
        model_output = list()
        actual_output = [1] * 8
        actual_output.extend([2] * 5)
        actual_output.extend([3] * 3)
        actual_output.extend([4] * 7)
        actual_output.extend([5] * 3)

        for sample in samples:
            outputs = self.forward_propagate(self.__best_Network, sample, bias,
                                             activation_function)
            f = outputs.index(max(outputs)) + 1
            if f == 1:
                model_output.append(1)
            elif f == 2:
                model_output.append(2)
            elif f == 3:
                model_output.append(3)
            elif f == 4:
                model_output.append(4)
            else:
                model_output.append(5)

        confusion_matrix = [[0 for x in range(5)]
                            for y in range(5)]

        for s in range(len(model_output)):
            y = actual_output[s]
            confusion_matrix[y - 1][model_output[s] - 1] += 1

        acc = (np.sum([confusion_matrix[i][i]
                       for i in range(5)]) / len(model_output)) * 100
        return acc