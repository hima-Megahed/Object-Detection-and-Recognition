import os
import cv2
import glob
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
import plotly
from plotly.graph_objs import *
from scipy.linalg import eigh
import numpy as np
from scipy import misc
from N_PCA import N_PCA

class TrainingData:
    def __init__(self):
        self.__TRAINING_PATH = "/home/harmoush/Downloads/Object-Detection-and-Recognition/Data set/Training"
        self.__TESTING_PATH = "/home/harmoush/Downloads/Object-Detection-and-Recognition/Data set/" \
                              "/Testing"
        # changing directory
        os.chdir(self.__TRAINING_PATH)
        self.__Training_Pics = [
            'Model1 - Cat.jpg',
            'Model2 - Cat.jpg',
            'Model3 - Cat.jpg',
            'Model4 - Cat.jpg',
            'Model5 - Cat.jpg',
            'Model6 - Laptop.jpg',
            'Model7 - Laptop.jpg',
            'Model8 - Laptop.jpg',
            'Model9 - Laptop.jpg',
            'Model10 - Laptop.jpg',
            'Model11 - Apple.jpg',
            'Model12 - Apple.jpg',
            'Model13 - Apple.jpg',
            'Model14 - Apple.jpg',
            'Model15 - Apple.jpg',
            'Model16 - Car.jpg',
            'Model17 - Car.jpg',
            'Model18 - Car.jpg',
            'Model19 - Car.jpg',
            'Model20 - Car.jpg',
            'Model21 - Helicopter.jpg',
            'Model22 - Helicopter.jpg',
            'Model23 - Helicopter.jpg',
            'Model24 - Helicopter.jpg',
            'Model25 - Helicopter.jpg'
        ]
        self.__TrainingData = np.ndarray(shape=(25, 2500), dtype=float)
        self.__PCA_TFeatures = np.ndarray(shape=(25, 2500), dtype=float)
        self.n_PCA = N_PCA()

    def read(self):
        # img = misc.imread(self.__TESTING_PATH + "\T1 - Cat Laptop .png")

        img_data = []
        # Reading Data From Training File
        for i in range(25):
            img = misc.imread(self.__Training_Pics[i], mode='L')
            # plt.imshow(img, cmap=plt.get_cmap('gray'))
            # plt.show()
            # Resizing Image to 50x50
            img = misc.imresize(img, (50, 50))
            # Converting Image from 2D to 1D
            img = np.reshape(img, 2500)

            if i < 5:
                img_data.append(img)
            elif i < 10:
                img_data.append(img)
            elif i < 15:
                img_data.append(img)
            elif i < 20:
                img_data.append(img)
            elif i < 25:
                img_data.append(img)

        # Scaling Data To Enhance Accuracy
        self.__TrainingData = StandardScaler().fit_transform(img_data)

        return self.__TrainingData

    def graph_pcs(self):
        """This function concerned with getting & graphing PCs"""
        # Getting Covariance matrix
        data_cov = np.cov(self.__TrainingData.T)
        # Getting Eigenvalues & EigenVectors
        eig_values, eig_vectors = eigh(data_cov)

        # Total Sum of Eigen Values
        tot = sum(eig_values)
        # Eigenvalues percentage in descending order
        var_exp = [(i / tot) * 100 for i in sorted(eig_values, reverse=True)]
        # Eigenvalues cumulative sum
        cum_var_exp = np.cumsum(var_exp)

        # explained variance for each Principle Component
        trace1 = Bar(
            x=['PC %s' % i for i in range(1, 30)],
            y=var_exp,
            showlegend=False)
        # cumulative sum for variance percentage over PC's
        trace2 = Scatter(
            x=['PC %s' % i for i in range(1, 30)],
            y=cum_var_exp,
            name='cumulative explained variance')

        data = Data([trace1, trace2])

        layout = Layout(
            yaxis=YAxis(title='Explained variance in percent'),
            title='Explained variance by different principal components')

        fig = Figure(data=data, layout=layout)
        # drawing The graph
        plotly.offline.plot(fig)

    def apply_pca(self):
        sklearn_pca = sklearnPCA(n_components=24)
        self.__PCA_TFeatures = sklearn_pca.fit_transform(self.__TrainingData)
        return self.__PCA_TFeatures

    def train_Neural_pca(self, samples):
      self.n_PCA.train_PCA(samples,0.03)

    def apply_Neural_pca(self,samples):
        pca_features=[]
        for sample in (samples):
            pca_features.append(self.n_PCA.Apply_PCA(sample))
        return pca_features


class TestingData:
    def __init__(self):
        self.__TestingData = list()
        self.__PCA_TestFeatures = list()
        self.TestingImages = [cv2.imread(file, 0) for file in glob.glob(
            "F:\GitHub - Projects\Object-Detection-and-Recognition\Data set\Custom Testing\*.png")]

    def read(self, TestingImages):
        tmp = []
        for img in TestingImages:
            # Resizing Image to 50x50
            img = misc.imresize(img, (50, 50))
            # Converting Image from 2D to
            img = np.reshape(img, 2500)
            tmp.append(img)
        self.__TestingData = StandardScaler().fit_transform(tmp)
        PCA_Features = self.__apply_pca()
        return PCA_Features

    def __apply_pca(self):
        sklearn_pca = sklearnPCA(n_components=24)
        self.__PCA_TestFeatures= sklearn_pca.fit_transform(self.__TestingData)
        return self.__PCA_TestFeatures

    def read_test_data_run(self):
        self.__PCA_TestFeatures = self.read(self.TestingImages)
        return self.__PCA_TestFeatures

