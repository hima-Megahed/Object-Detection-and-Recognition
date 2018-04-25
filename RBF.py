from Clustering import K_Means
import numpy as np
import math

class RadialBasisFunction:
    def __init__(self,features,learnRate,numEpochs,threshold,numHiddenNeurons):
        self.features = features
        self.learnRate = learnRate
        self.numEpochs = numEpochs
        self.threshold = threshold
        self.numHiddenNeurons = numHiddenNeurons
        self.clusteringData = K_Means(features,self.numHiddenNeurons)
        self.centroids = K_Means.Centroids

    def mseTrain(self):
        epoch = 0
        weights = [[np.random.rand(1)[0] in range(self.numHiddenNeurons)] for i in range(5)]
        for w in weights:
            print(w)

        while epoch < self.numEpochs :
            for i in range(0, len(self.features)):
                #classlabel = self.get_sample_class(i)
                d_value = self.get_sample_class(i)
                d = np.zeros(5)
                d[d_value-1] = 1
                x = self.features[i]
                v = [0 in range(5)]
                for output_neuron in range(5):
                    w = weights[output_neuron]
                    v[output_neuron] = self.net_input(x, w)
                    y = v
                    y = np.asarray(y)
                    error = d - y
                    for w_index in range(len(w)):
                        w[w_index] = w[w_index] + self.learnRate * error[output_neuron] * w[w_index]


            epoch = epoch + 1
            errorMSE = self.updateError(weights,self.features)
            if errorMSE < self.threshold:
                break

        return weights

    def updateError(self, weights,features):
        MSE = 0
        for i in range(0, len(features)):
            d_value = self.get_sample_class(i)
            d = np.zeros(5)
            d[d_value - 1] = 1
            x = self.features[i]
            v = [0 in range(5)]
            MSE_ = 0
            for output_neuron in range(5):
                w = weights[output_neuron]
                v[output_neuron] = self.net_input(x, w)
                y = v
                y = np.asarray(y)
                error = d - y
                MSE_ = MSE_+(error ** 2)
            MSE = MSE + MSE_
        MSE = MSE / (2 * len(features))
        return MSE

    def mseTest(self,features, weights):
        v = [0 in range(5)]
        for output_neuron in range(5):
            w = weights[output_neuron]
            v[output_neuron] = self.net_input(features, w)
        y = np.argmax(self.softmax(v))
        return y + 1

    def net_input(self, x, weight):
        """Calculate net input"""
        v = 0
        for i in range(len(x)):
            v = v + x[i]* weight[i]
        return v

    def get_sample_class(self,num):
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

    def update_features(self):
        sigma,max_distance = self.compute_sigma()
        new_features = [[0 in range(self.features)] for i in range(self.numHiddenNeurons)]
        for i in range(len(new_features)):
            for j in range(0,len(self.features[i])):
                r_square = K_Means.EculideanDistance(self.features[j],self.centroids[i]) ** 2
                double_sigma_square = 2 * (sigma ** 2)
                new_features[i][j] = math.exp(-1 * (r_square/double_sigma_square))

        self.features = new_features


    def compute_sigma(self):
        max_distance = -1
        d = -1
        for i in range(0,self.numHiddenNeurons):
            for j in range (0,self.numHiddenNeurons):

                if i == j:
                    continue

                K_Means.EculideanDistance(self.centroids[i],self.centroids[j])
                if d > max_distance:
                    max_distance = d
        sigma = max_distance/math.sqrt(2 * self.numHiddenNeurons)
        return sigma,max_distance

    def softmax(v):
        mean = np.mean(v)
        std = np.std(v)
        norm = (v - mean) / std
        e_v = np.exp(norm)
        return e_v / e_v.sum