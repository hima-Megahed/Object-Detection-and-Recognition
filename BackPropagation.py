import numpy as np
import math
import matplotlib.pyplot as plt
from DataManipulation import TrainingData


class BackPropagation:

    def __init__(self):
        self.weights = {}
        self.weights_inputs = {}
        self.errors = []

    def main_algorithm(self, features, eta, epochs, bias, threshold,
                       stopping_criteria, activation_function,
                       num_hidden_layer, num_neurons_layer, n_samples):
        """This function will run the back propagation algorithm"""

        # Initializing weight vector with random values
        weights, weights_inputs = self.initialize(num_hidden_layer,
                                                  num_neurons_layer)

        if stopping_criteria == 1:  # Number of epochs
            # loop through number of epochs
            for i in range(epochs):
                sample_ind = 1
                # loop through number of samples
                for sample in features:
                    # getting input vector
                    input_vector = sample
                    desired_output = self.__get_sample_class(sample_ind)
                    weights_inputs = self.net_input(input_vector,
                                                    weights, weights_inputs,
                                                    bias, num_hidden_layer,
                                                    num_neurons_layer,
                                                    activation_function)
                    error = self.propagate_error(weights_inputs, weights,
                                                 num_hidden_layer,
                                                 num_neurons_layer,
                                                 desired_output,
                                                 activation_function)

                    weights = self.update_weights(weights_inputs, weights,
                                                  num_hidden_layer,
                                                  num_neurons_layer, eta,
                                                  error, bias, input_vector)

                    self.weights = weights
                    self.errors = error
                    self.weights_inputs = weights_inputs
                    sample_ind += 1

        else:  # Threshold MSE
            epoch_ind = 1
            mse = 10000000.0
            mse_errors = [0] * 5

            while epoch_ind < 1000:
                sample_ind = 1
                # loop through number of samples
                for sample in features:
                    # getting input vector
                    input_vector = sample
                    desired_output = self.__get_sample_class(sample_ind)
                    weights_inputs = self.net_input(input_vector,
                                                    weights, weights_inputs,
                                                    bias, num_hidden_layer,
                                                    num_neurons_layer,
                                                    activation_function)
                    error = self.propagate_error(weights_inputs, weights,
                                                 num_hidden_layer,
                                                 num_neurons_layer,
                                                 desired_output,
                                                 activation_function)

                    weights = self.update_weights(weights_inputs, weights,
                                                  num_hidden_layer,
                                                  num_neurons_layer, eta,
                                                  error, bias, input_vector)

                    self.weights = weights
                    self.errors = error
                    self.weights_inputs = weights_inputs
                    sample_ind += 1

                mse, mse_errors = self.compute_mean_square_error(
                    self.errors[-5:], mse_errors, n_samples)
                epoch_ind += 1
                if mse <= threshold:
                    po = 0
                    break
            pi = 4

            #return MSE_arr, epochs_arr

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
                    vsum = 0.0  # sum of weights in single node
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
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

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
            return self.Hyperbolic_tangent_derivative(x)

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1-x)

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
    def update_weights(weights_inputs, weight, num_hidden_layer
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

    def main_algorithm_testing(self, features, bias, activation_function,
                               num_hidden_layer,num_neurons_layer):
        Output = []
        for j in range(len(features["X1"])):
                    # getting input vector
                    X = [features["X1"][j], features["X2"][j],
                         features["X3"][j], features["X4"][j]]
                    YOut = features["Y"][j]
                    weights_inputs = self.NetInput(X, self.weights, self.weights_inputs,
                                                   bias, num_hidden_layer,
                                                   num_neurons_layer,
                                                   activation_function)
                    Length = len(weights_inputs)
                    if weights_inputs[Length - 1] > weights_inputs[Length - 2] and \
                        weights_inputs[Length - 1] > weights_inputs[Length -3]:
                        Output.append(1)
                    elif weights_inputs[Length - 2] > weights_inputs[Length - 1] and \
                        weights_inputs[Length - 2] > weights_inputs[Length -3]:
                        Output.append(2)
                    else:
                        Output.append(3)
        return Output

    @staticmethod
    def __get_sample_class(num):
        if 1 <= num <= 5:
            return 1
        elif 6 <= num <= 10:
            return 2
        elif 11 <= num <= 15:
            return 3
        elif 16 <= num <= 20:
            return 4
        else:
            return 5
