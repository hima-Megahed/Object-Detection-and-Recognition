import numpy as np
import math
import matplotlib.pyplot as plt
from DataManipulation import TrainingData


class BackPropagation:

    def __init__(self, num_neurons_layer, num_hidden_layer):
        # Initializing weight vector with random values
        self.Network = self.initialize_network(num_neurons_layer,
                                               num_hidden_layer)

    def main_algorithm(self, features, eta, epochs, bias, threshold,
                       stopping_criteria, activation_function,
                       num_hidden_layer, num_neurons_layer, n_samples):
        """This function will run the back propagation algorithm"""



        # Number of epochs
        if stopping_criteria == 1:
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
                print('> epoch=%d, error=%.5f, Minimum is: %.5f'% (i, sum_error, minsum))
        # Threshold MSE
        elif stopping_criteria == 2:
            epoch_ind = 1
            mse = 10000000.0
            mse_errors = [0] * 5

            while epoch_ind < 1000:
                sample_ind = 1
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
                    self.errors[-5:], mse_errors, n_samples)
                epoch_ind += 1
                if mse <= threshold:
                    po = 0
                    break
            pi = 4

            #return MSE_arr, epochs_arr
        # Cross Validation
        else:
            # loop through number of epochs
            for i in range(epochs):
                sample_ind = 0
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
                    sample_ind += 1

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
                    print(acc, '%')

                    asdf = 0

    def net_input(self, input_vector, weight, weights_inputs, bias,
                  num_hidden_layer, num_neurons_layer, activation_function):
        """This function will get the Net of each neuron """

        ind = 1
        ind_w_input = 0
        # num of hidden layers and output layer
        for i in range(num_hidden_layer + 1):
            # First Layer of the hidden layers
            if i == 0:
                # Number of neurons in first layer of hidden layers
                for j in range(num_neurons_layer):
                    vsum = 0  # sum of weights in single node
                    # Number of weights 25 in the node in first layer of
                    # hidden layers 24 + 1 bias
                    for a in range(0, 25):
                        # bias
                        if a == 0:
                            vsum += bias * weight["w"+str(ind)][a]
                        # Rest of weights in single node
                        else:
                            vsum += input_vector[a-1] * weight["w"+str(ind)][a]

                    y = self.activate(activation_function, vsum)
                    weights_inputs[ind-1] = y
                    ind += 1
            # Output Layer
            elif i == num_hidden_layer:
                # Number of Nodes in output layer
                for j in range(5):
                    vsum = 0.0  # sum of weights in single node
                    # Number of weights in single node
                    for a in range(0, num_neurons_layer + 1):
                        # bias
                        if a == 0:
                            vsum += bias * weight["w" + str(ind)][a]
                        # Rest of weights in single node
                        else:
                            vsum += weights_inputs[ind_w_input + a - 1] * \
                                    weight["w" + str(ind)][a]

                    y = self.activate(activation_function, vsum)
                    weights_inputs[ind-1] = y
                    ind += 1
            # Regular Layer in th hidden layer
            else:
                # Number of neurons in regular layer in hidden layers
                for j in range(num_neurons_layer):
                    vsum = 0.0  # sum of weights in single node
                    # Number of weights in single node
                    for a in range(0, num_neurons_layer + 1):
                        # bias
                        if a == 0:
                            vsum += bias * weight["w" + str(ind)][a]
                        # Rest of weights in single node
                        else:
                            vsum += weights_inputs[ind_w_input + a - 1] * \
                                    weight["w" + str(ind)][a]

                    y = self.activate(activation_function, vsum)
                    weights_inputs[ind-1] = y
                    ind += 1
                ind_w_input += num_neurons_layer
        return weights_inputs

    @staticmethod
    def compute_mean_square_error(error, mse_errors, n_samples):
        err_sum = 0.0
        tmp = []
        for i in range(len(error)):
            mse_errors[i] += error[i]**2

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

    def activate(self,activation_function, x):
        if activation_function == 1:
            return self.sigmoid(x)
        else:
            return self.hyperbolic_tangent(x)

    @staticmethod
    def initialize(num_hidden_layer, num_neurons_layer):
        weights = {}
        weights_inputs = []
        ind = 1

        # num of hidden layers and output layer
        for i in range(num_hidden_layer + 1):
            # First Layer have 24 weights to every node in the first layer
            # of the hidden layer + 1 bias
            if i == 0:
                for j in range(num_neurons_layer):
                    weights["w" + str(ind)] = [np.random.rand(1)[0]
                                               for k
                                               in range
                                               (25)]
                    weights_inputs.append(0)
                    ind += 1
            # Output Layer 5 nodes ONLY
            elif i == num_hidden_layer:
                for j in range(int(5)):
                    # inputs to neuron is number of neurons per layer + bias
                    weights["w" + str(ind)] = [np.random.rand(1)[0]
                                               for k
                                               in range
                                               (num_neurons_layer + 1)]
                    weights_inputs.append(0)
                    ind += 1
            # Hidden Layers in the middle Not First neither output
            else:
                # inputs to neuron is number of neurons per layer + bias
                for j in range(int(num_neurons_layer)):
                    weights["w" + str(ind)] = [np.random.rand(1)[0]
                                               for k
                                               in range
                                               (num_neurons_layer + 1)]
                    weights_inputs.append(0)
                    ind += 1
    # TODO: Implement backward error propagation
        return weights, weights_inputs

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

    def propagate_error(self, weights_inputs, weight, num_hidden_layer,
                        num_neurons_layer, desired_output,
                        activation_function):

        error = [0] * len(weights_inputs)
        ind = len(weights_inputs) - 1
        # Number of Hidden layers + Output layer,
        #  starting from output layer
        for i in reversed(range(num_hidden_layer + 1)):
            # Regular Layer in the hidden layers
            if i != num_hidden_layer:
                for j in range(num_neurons_layer):
                    vsum = 0.0  # sum of weights in single node
                    # The Layer that directly precede output layer
                    if i + 1 == num_hidden_layer:
                        # Number of Neurons in output layer
                        for k in range(5):
                            vsum += weights_inputs[
                                        (i + 1) * num_neurons_layer + k] \
                                    * error[(i + 1) * num_neurons_layer + k]
                    # Rest Of Layers
                    else:
                        # Number of neurons in each layer in Hidden layers
                        for k in range(num_neurons_layer):
                            vsum += weights_inputs[
                                        (i + 1) * num_neurons_layer + k] \
                                    * error[(i + 1) * num_neurons_layer + k]

                    error[ind] = vsum * self.derivative_transfer(
                        activation_function, weights_inputs[ind])
                    ind -= 1
            # Output Layer
            else:
                # Number of neurons in the Output Layer
                for j in range(5):
                    y = desired_output - weights_inputs[ind]
                    error[ind] = y * self.derivative_transfer(
                        activation_function, weights_inputs[ind])
                    ind -= 1

        return error

    @staticmethod
    def update_weights1(weights_inputs, weight, num_hidden_layer
                       , num_neurons_layer, eta, error, bias, input_vector):
        ind = len(weight)
        # Number of neurons in output layer + 1 because array is 0 based
        ind_w_input = len(weights_inputs) - 6
        ind_e = len(error) - 1
        # Number of Hidden Layers + Output Layer
        for i in reversed(range(num_hidden_layer + 1)):
            # Output layer
            if i == num_hidden_layer:
                # Number of Neurons in Output layer
                for j in reversed(range(5)):
                    # Number of weights from neurons in each layer + bias
                    for k in reversed(range(num_neurons_layer + 1)):
                        # weight from Bias
                        if k == 0:
                            weight["w" + str(ind)][k] = eta * bias\
                                                     * error[ind_e]
                        # weight from Regular neuron
                        else:
                            weight["w" + str(ind)][k] = eta * \
                                                        weights_inputs[
                                                            ind_w_input]\
                                                        * error[ind_e]
                            ind_w_input -= 1
                    ind -= 1
                    ind_w_input += num_neurons_layer
                    ind_e -= 1
            # First Layer
            elif i == 0:
                # Number of neurons in first layer of the Hidden layers
                for j in reversed(range(num_neurons_layer)):
                    # Number of weights from Input layer 24 + bias for
                    # each neuron
                    for k in reversed(range(25)):
                        if k == 0:
                            weight["w" + str(ind)][k] = eta * bias\
                                                     * error[ind_e]
                        else:
                            weight["w" + str(ind)][k] = eta * \
                                                        input_vector[k - 1] \
                                                        * error[ind_e]
                    ind -= 1
                    ind_e -= 1
            # Regular Layers in Hidden layers
            else:
                ind_w_input -= num_neurons_layer
                # Number of neurons in Regular layer in the Hidden layers
                for j in range(num_neurons_layer):
                    # Number of weights from neurons in the layer + Bias
                    for k in reversed(range(num_neurons_layer + 1)):
                        # Bias
                        if k == 0:
                            weight["w" + str(ind)][k] = eta * bias\
                                                     * error[ind_e]
                        # Weights from rest of neurons
                        else:
                            weight["w" + str(ind)][k] = eta * \
                                                        weights_inputs[
                                                            ind_w_input]\
                                                        * error[ind_e]
                            ind_w_input -= 1
                    ind -= 1
                    ind_w_input += num_neurons_layer
                    ind_e -= 1
        return weight

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
            activation += weights[i] * inputs[i-1]
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

    def update_weights(self, network, data_row, l_rate, bias):
        for i in range(len(network)):
            inputs = data_row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(1, len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] \
                                            * inputs[j]
                neuron['weights'][0] += l_rate * neuron['delta'] * bias
