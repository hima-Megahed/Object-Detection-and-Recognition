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
        self.centroids = self.clusteringData.Centroids

    def mseTrain(self):
        min_mse = 100000
        epoch = 0
        #weights = [[np.random.rand(1)[0] in range(self.numHiddenNeurons)] for i in range(5)]
        weights = [[] for i in range(5)]
        for i in range(5):
            for j in range(self.numHiddenNeurons):
                x = np.random.rand(1)[0]
                weights[i].append(x)
            #weights[i] = [x for j in range(self.numHiddenNeurons)]
        # for w in weights:
            # print(w)
        self.update_features()
        num_samples = len(self.features)
        while epoch < self.numEpochs :
            for i in range(num_samples):
                #classlabel = self.get_sample_class(i)
                d_value = self.get_sample_class(i)
                d = np.zeros(5)
                d[d_value-1] = 1
                x = self.features[i]
                v = [0 for i in range(5)]
                for output_neuron in range(5):
                    w = weights[output_neuron]
                    v[output_neuron] = self.net_input(x, w)
                y = v
                y = np.asarray(y)
                error = d - y
                lastweights = weights
                for output_neuron in range(5):
                    w = weights[output_neuron]
                    for w_index in range(len(w)):
                        w[w_index] = w[w_index] + self.learnRate * error[output_neuron] * w[w_index]
                    weights[output_neuron] = w


            epoch = epoch + 1
            errorMSE = self.updateError(weights,self.features)
            if min_mse > errorMSE:
                min_mse = errorMSE
            print(">Epoch:{}, Mean Square Error:{}, Min MSE:{}".format(epoch, errorMSE, min_mse))
            if errorMSE < self.threshold and epoch > 100:
                break

        return weights,self.centroids

    def updateError(self, weights,features):
        average_mse = 0
        MSE = [0 for i in range(5)]
        num_samples = len(features)
        for i in range(num_samples):
            d_value = self.get_sample_class(i)
            d = np.zeros(5)
            d[d_value - 1] = 1
            x = self.features[i]
            v = [0 for i in range(5)]

            for output_neuron in range(5):
                w = weights[output_neuron]
                v[output_neuron] = self.net_input(x, w)
            y = v
            y = np.asarray(y)
            error = d - y
            for output_neuron in range(5):
                MSE[output_neuron] = MSE[output_neuron]+(error[output_neuron] ** 2)
        for output_neuron in range(5):
            MSE[output_neuron] = MSE[output_neuron] / (2 * num_samples)
        average_mse = np.sum(MSE) / 5
        return average_mse

    def mseTest(self,features, weights,centroids):
        literal_output = list()
        for sample in features:
            v = [0 for i in range(5)]
            self.features = sample
            self.centroids = centroids
            self.update_features_test()
            for output_neuron in range(5):
                w = weights[output_neuron]
                v[output_neuron] = self.net_input(self.features, w)
            probabilities = self.softmax(v)
            #sum = np.sum(probabilities)
            f = np.argmax(probabilities) + 1
            if f == 1:
                literal_output.append("Cat")
            elif f == 2:
                literal_output.append("Laptop")
            elif f == 3:
                literal_output.append("Apple")
            elif f == 4:
                literal_output.append("Car")
            else:
                literal_output.append("Helicopter")

        return literal_output

    def net_input(self, x, weight):
        """Calculate net input"""
        v = 0
        for i in range(len(x)):
            v = v + x[i]* weight[i]
        return v

    def get_sample_class(self,num):
        if 0 <= num <= 4:
            return 1
        elif 5 <= num <= 9:
            return 2
        elif 10 <= num <= 14:
            return 3
        elif 15 <= num <= 19:
            return 4
        else:
            return 5

    def update_features(self):
        sigma,max_distance = self.compute_sigma()
        num_samples = len(self.features)
        new_features = [[0 for i in range(self.numHiddenNeurons)] for i in range(num_samples)]
        for i in range(num_samples):
            for j in range(0,self.numHiddenNeurons):
                r_square = self.EculideanDistance(self.features[i],self.centroids[j]) ** 2
                double_sigma_square = 2 * (sigma ** 2)
                mo= math.exp(-1 * (r_square/double_sigma_square))
                new_features[i][j] = math.exp(-1 * (r_square/double_sigma_square))

        self.features = new_features

    def update_features_test(self):
        sigma,max_distance = self.compute_sigma()
        new_features = [0 for i in range(self.numHiddenNeurons)]

        for j in range(0,self.numHiddenNeurons):
            r_square = self.EculideanDistance(self.features,self.centroids[j]) ** 2
            double_sigma_square = 2 * (sigma ** 2)
            mo= math.exp(-1 * (r_square/double_sigma_square))
            new_features[j] = math.exp(-1 * (r_square/double_sigma_square))

        self.features = new_features

    def compute_sigma(self):
        max_distance = -1
        d = -1
        for i in range(0,self.numHiddenNeurons):
            for j in range (0,self.numHiddenNeurons):

                if i == j:
                    continue

                d = self.EculideanDistance(self.centroids[i],self.centroids[j])

                if d > max_distance:
                    max_distance = d
        sigma = max_distance/math.sqrt(2 * self.numHiddenNeurons)
        return sigma,max_distance

    def softmax(self,v):
        mean = np.mean(v)
        std = np.std(v)
        norm = (v - mean) / std
        e_v = np.exp(norm)
        sum = np.sum(e_v)
        return (e_v / sum)
    def EculideanDistance(self,Feature, Centroid):
        SquaredDistance = 0.0
        for i in range(len(Centroid)):
            SquaredDistance += (Centroid[i] - Feature[i])**2
            # print(Centroid[i] , " " , Feature[i] ,"SquaredDistance = " ,SquaredDistance ,end=" \n")
        # print(SquaredDistance,"  ",math.sqrt(SquaredDistance))

        return math.sqrt(SquaredDistance)