import numpy as np
import math
import matplotlib.pyplot as plt
from DataManipulation import TrainingData


class BackPropagation:

    def __init__(self):
        self.weights = {}
        self.weights_inputs = {}

    def main_algorithm(self, features, eta, epochs, bias, threshold,
                       stopping_criteria, activation_function,
                       num_hidden_layer, num_neurons_layer):
        """This function will run the back propagation algorithm"""

        # Initializing weight vector with random values
        weights, weights_inputs = self.initialize(num_hidden_layer, num_neurons_layer)

        if stopping_criteria == 1:  # Number of epochs
            # loop through number of epochs
            for i in range(epochs):
                # loop through number of samples
                for j in range(len(features["X1"])):
                    # getting input vector
                    X = [features["X1"][j], features["X2"][j],
                         features["X3"][j], features["X4"][j]]
                    YOut = features["Y"][j]
                    weights_inputs = self.NetInput(X, weights, weights_inputs,
                                                   bias, num_hidden_layer,
                                                   num_neurons_layer,
                                                   activation_function)
                    error = self.propagate_error(weights_inputs, weights,
                                                 num_hidden_layer,
                                                 num_neurons_layer,
                                                 YOut, activation_function)

                    weights = self.update_weights(weights_inputs, weights, num_hidden_layer
                       ,num_neurons_layer, eta, error, bias, X)
                    self.weights = weights
                    self.weights_inputs = weights_inputs

        else:  # Threshold MSE
            MSE_arr = []
            epochs_arr = []
            Epoch_Ind = 1
            MSE = 10000000.0
            while MSE > threshold and Epoch_Ind < 1000:
               for j in range(len(features["X1"])):
                    # getting input vector
                    X = [features["X1"][j], features["X2"][j],
                         features["X3"][j], features["X4"][j]]
                    YOut = features["Y"][j]
                    weights_inputs = self.NetInput(X, weights, weights_inputs,
                                                   bias, num_hidden_layer,
                                                   num_neurons_layer,
                                                   activation_function)
                    error = self.propagate_error(weights_inputs, weights,
                                                 num_hidden_layer,
                                                 num_neurons_layer,
                                                 YOut, activation_function)

                    weights = self.update_weights(weights_inputs, weights, num_hidden_layer
                       ,num_neurons_layer, eta, error, bias, X)

                    MSE = self.ComputeMeanSquareError(error)
                    MSE_arr.append(MSE)
                    epochs_arr.append(Epoch_Ind)
                    Epoch_Ind += 1
                    print(MSE)
            return MSE_arr, epochs_arr
            # p.PlotIris()
            # plt.show()

    def net_input(self, X, weight, weights_inputs, bias, num_hidden_layer,
                  num_neurons_layer, activation_function):
        """This function will get the Net of each neuron """
        ind = 1
        ind_WInput = 0
        for i in range(num_hidden_layer + 1):
            if i == 0:
                for j in range(num_neurons_layer):
                    V = bias * weight["w"+str(ind)][0] \
                        + X[0] * weight["w"+str(ind)][1] \
                        + X[1] * weight["w"+str(ind)][2] \
                        + X[2] * weight["w"+str(ind)][3] \
                        + X[3] * weight["w"+str(ind)][4]

                    Y = self.activate(activation_function, V)
                    weights_inputs[ind-1] = Y
                    ind += 1
            elif i == num_hidden_layer:
                for j in range(3):
                    V = bias * weight["w"+str(ind)][0]
                    for k in range(1, num_neurons_layer + 1):
                        V += weights_inputs[ind_WInput + k - 1] * weight["w"+str(ind)][k]

                    Y = self.activate(activation_function, V)
                    weights_inputs[ind-1] = Y
                    ind += 1
            else:
                for j in range(num_neurons_layer):
                    V = bias * weight["w" + str(ind)][0]
                    for k in range(1, num_neurons_layer + 1):
                        V += weights_inputs[ind_WInput + k - 1] * \
                             weight["w" + str(ind)][k]
                    Y = self.activate(activation_function, V)
                    weights_inputs[ind-1] = Y
                    ind += 1
                ind_WInput += num_neurons_layer
        return weights_inputs

    @staticmethod
    def compute_mean_square_error(error):
        sum = 0.0
        for i in range(len(error)):
            sum += error[i]**2
        return sum/len(error)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def hyperbolic_tangent(x):
        return np.tanh(x)

    def activate(self,activation_function, x):
        if activation_function == 1:
            return self.sigmoid(x)
        else:
            return self.Hyperbolic_tangent(x)

    @staticmethod
    def initialize(num_hidden_layer, num_neurons_layer):
        weights = {}
        weights_inputs = []
        ind = 1
        for i in range(num_hidden_layer + 1):  # num of hidden layers and output layer
            if i == 0:
                for j in range(num_neurons_layer):
                    weights["w" + str(ind)] = [np.random.rand(1)[0]
                                               for k
                                               in range
                                               (5)]
                    weights_inputs.append(0)
                    ind += 1
            elif i == num_hidden_layer:
                for j in range(int(3)):
                    weights["w" + str(ind)] = [np.random.rand(1)[0]
                                               for k
                                               in range
                                               (
                                                   num_neurons_layer + 1)]  # inputs to neuron number of last neurons + bias
                    weights_inputs.append(0)
                    ind += 1
            else:
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
                        num_neurons_layer, YOut, activation_function):
        error = [0] * len(weights_inputs)
        ind = len(weights_inputs) - 1
        for i in reversed(range(num_hidden_layer+1)):
            if i != num_hidden_layer:
                for j in range(num_neurons_layer):
                    sum = 0.0
                    if i+1 == num_hidden_layer:
                        for k in range(3):
                            sum += weights_inputs[(i+1)*num_neurons_layer+k]*error[(i+1)*num_neurons_layer+k]
                    else:
                        for k in range(num_neurons_layer):
                            sum += weights_inputs[(i + 1) * num_neurons_layer + k]*error[(i+1)*num_neurons_layer+k]

                    error[ind] = sum * self.derivative_transfer(activation_function, weights_inputs[ind])
                    ind -= 1

            else:
                for j in range(3):
                    y = YOut - weights_inputs[ind]
                    error[ind]=y* self.derivative_transfer(activation_function, weights_inputs[ind])
                    ind -= 1

        return error

    @staticmethod
    def update_weights(weights_inputs, weight, num_hidden_layer
                       , num_neurons_layer, eta, error, bias, X):
        ind = len(weight)
        ind_WInput = len(weights_inputs) - 4
        ind_E = len(error) - 1
        for i in reversed(range(num_hidden_layer + 1)):
            # output layer weights
            if i == num_hidden_layer:
                for j in reversed(range(3)):
                    for k in reversed(range(num_neurons_layer + 1)):
                        if k == 0:
                            weight["w" + str(ind)][k] = eta * bias\
                                                     * error[ind_E]
                        else:
                            weight["w" + str(ind)][k] = eta * weights_inputs[ind_WInput] \
                                                     * error[ind_E]
                            ind_WInput -= 1
                    ind -= 1
                    ind_WInput += num_neurons_layer
                    ind_E -= 1
            elif i == 0:
                for j in reversed(range(num_neurons_layer)):
                    for k in reversed(range(5)):
                        if k == 0:
                            weight["w" + str(ind)][k] = eta * bias\
                                                     * error[ind_E]
                        else:
                            weight["w" + str(ind)][k] = eta * X[k-1] \
                                                     * error[ind_E]
                    ind -= 1
                    ind_E -= 1
            else:
                ind_WInput -= num_neurons_layer

                for j in range(num_neurons_layer):
                    for k in reversed(range(num_neurons_layer + 1)):
                        if k == 0:
                            weight["w" + str(ind)][k] = eta * bias\
                                                     * error[ind_E]
                        else:
                            if ind_E < 0:
                                print("err ", ind_E)
                            weight["w" + str(ind)][k] = eta * weights_inputs[ind_WInput]\
                                                        * error[ind_E]
                            ind_WInput -= 1
                    ind -= 1
                    ind_WInput += num_neurons_layer
                    ind_E -= 1
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
