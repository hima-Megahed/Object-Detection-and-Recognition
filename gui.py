import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from readData import ReadData
from NNTask3.plotIris import PlotIris
from NNTask3.BackPropagation import BackPropagation
from NNTask3.AdaLineAlgorithm import AdaLineAlgo
import numpy as np


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.choosenClasses= tk.StringVar(self.root)
        self.chosenFeatures = tk.StringVar(self.root)
        self.learnRate = tk.StringVar(self.root)
        self.learnRate.set(0.01)
        self.epochsNo = tk.StringVar(self.root)
        self.epochsNo.set(4)
        self.bias = tk.IntVar(self.root)
        self.bias.set(0)
        self.root.resizable(height=True, width=True)
        self.root.title("NN Task3")
        self.root.geometry("500x500")
        self.training_features = {
            'X1': [],
            'X2': [],
            'X3': [],
            'X4': [],
            'Y': []
        }
        self.testing_features = {
            'X1': [],
            'X2': [],
            'X3': [],
            'X4': [],
            'Y': []
        }
        self.dataset = []
        self.errorTh =tk.DoubleVar(self.root)
        self.errorTh.set(0.001)
        self.plotLine = tk.IntVar(self.root)
        self.plotLine.set(0)
        self.c1 = 0
        self.c2 = 0
        self.w1 = 0
        self.w2 = 0
        self.b = 0
        self.mainAlgo = []
        self.NumberOfHiddenLayers = tk.IntVar(self.root)
        self.NumberOfHiddenLayers.set(3)
        self.NumberOfNeuronsInEachLayer = tk.IntVar(self.root)
        self.NumberOfNeuronsInEachLayer.set(5)
        self.activationFunction = tk.IntVar()
        self.activationFunction.set(1)
        self.stoppingCriteria = tk.IntVar()
        self.stoppingCriteria.set(1)
        self.initializeComponents()
        self.root.mainloop()
        #######################

    def initializeComponents(self):
        def defocus(event):
            event.widget.master.focus_set()
        tk.Label(self.root,text="# Of Hidden Layers").place(relx=0.11, rely=0.05)
        HLEntry = tk.Entry(self.root,width=10 , textvariable = self.NumberOfHiddenLayers)
        HLEntry.place(relx=0.64, rely=0.05)

        tk.Label(self.root,text="# Of Neurons In Each Layer").place(relx=0.11, rely=0.14)
        NEntry = tk.Entry(self.root, width=10, textvariable = self.NumberOfNeuronsInEachLayer)
        NEntry.place(relx=0.64, rely=0.14)

        tk.Label(self.root,text="Enter Learning Rate(eta):").place(relx=0.11, rely=0.23)
        eta = tk.Entry(self.root,width=10,textvariable=self.learnRate )
        eta.place(relx=0.64, rely=0.23)

        tk.Label(self.root,text="Enter Number of Epochs:").place(relx=0.11, rely=0.32)
        epochs = tk.Entry(self.root,width=10,textvariable=self.epochsNo)
        epochs.place(relx=0.64, rely=0.32)

        tk.Label(self.root, text="Choose an activation function").place(relx=0.03, rely=0.41)
        tk.Radiobutton(self.root,
                       text="Sigmoid",
                       variable=self.activationFunction,
                       value=1).place(relx=0.03, rely=0.45)
        tk.Radiobutton(self.root,
                       text="Hyperbolic Tangent Sigmoid",
                       variable=self.activationFunction,
                       value=2).place(relx=0.03, rely=0.49)

        tk.Label(self.root, text="Bias:").place(relx=0.70, rely=0.41)
        tk.Checkbutton(self.root, variable=self.bias).place(relx=0.78,
                                                            rely=0.41)

        tk.Label(self.root, text="Choose The Stopping Criteria").place(
            relx=0.03, rely=0.58)
        tk.Radiobutton(self.root,
                       text="# of Epochs",
                       variable=self.stoppingCriteria,
                       value=1).place(relx=0.03, rely=0.62)
        tk.Radiobutton(self.root,
                       text="MSE Threshold",
                       variable=self.stoppingCriteria,
                       value=2).place(relx=0.03, rely=0.66)

        tk.Entry(self.root, width=10, textvariable=self.errorTh)\
            .place(relx=0.64, rely=0.66)

        tk.Button(self.root, text="Plotting",width=10, fg="Black",bg="light Gray", command=lambda: self.plotFeatures()).place(relx=0.03, rely=0.80)
        tk.Button(self.root, text="Learning",width=10, fg="Black",bg="light Gray", command=lambda: self.learning()).place(relx=0.35, rely=0.80)
        tk.Button(self.root, text="Testing",width=10, fg="Black",bg="light Gray", command=lambda: self.testing()).place(relx=0.67, rely=0.80)
        
    def manageTrainingFeatures(self):
        #initilize X1 & X2 & X3 & X4 & Y
        rd = ReadData()
        rd.readData()

        # Reading first chunk of data
        self.training_features['X1'] = rd.IrisX1[0:30]
        self.training_features['X2'] = rd.IrisX2[0:30]
        self.training_features['X3'] = rd.IrisX3[0:30]
        self.training_features['X4'] = rd.IrisX4[0:30]
        self.training_features['Y'] = [1 for i in range(0, 30)]
        # Reading second chunk of data
        self.training_features['X1'].extend(rd.IrisX1[50:80])
        self.training_features['X2'].extend(rd.IrisX2[50:80])
        self.training_features['X3'].extend(rd.IrisX3[50:80])
        self.training_features['X4'].extend(rd.IrisX4[50:80])
        self.training_features['Y'].extend([2 for i in range(50, 80)])
        # Reading third chunk of data
        self.training_features['X1'].extend(rd.IrisX1[100:130])
        self.training_features['X2'].extend(rd.IrisX2[100:130])
        self.training_features['X3'].extend(rd.IrisX3[100:130])
        self.training_features['X4'].extend(rd.IrisX4[100:130])
        self.training_features['Y'].extend([3 for i in range(100, 130)])




        self.testing_features['X1'] = rd.IrisX1[30:50]
        self.testing_features['X2'] = rd.IrisX2[30:50]
        self.testing_features['X3'] = rd.IrisX3[30:50]
        self.testing_features['X4'] = rd.IrisX4[30:50]
        self.testing_features['Y'] = [1 for i in range(30,50)]
        # Reading second chunk of data
        self.testing_features['X1'].extend(rd.IrisX1[80:100])
        self.testing_features['X2'].extend(rd.IrisX2[80:100])
        self.testing_features['X3'].extend(rd.IrisX3[80:100])
        self.testing_features['X4'].extend(rd.IrisX4[80:100])
        self.testing_features['Y'].extend([2 for i in range(80,100)])
        # Reading third chunk of data
        self.testing_features['X1'].extend(rd.IrisX1[130:150])
        self.testing_features['X2'].extend(rd.IrisX2[130:150])
        self.testing_features['X3'].extend(rd.IrisX3[130:150])
        self.testing_features['X4'].extend(rd.IrisX4[130:150])
        self.testing_features['Y'].extend([3 for i in range(130, 150)])



    def manageTestingFeatures(self):
        featureX,featureY = self.initilizeData()
        choosenClasses = self.choosenClasses.get()
        class1 = int(choosenClasses[1])
        class2 = int(choosenClasses[6])
        self.c1 = class1
        self.c2 = class2
        ####
        if class1 == 1:
            self.X1_testing = featureX[30:50]
            self.X2_testing = featureY[30:50]
            self.classlable_testing = [1 for i in range(0,20)]
            if class2 == 2:
                self.X1_testing.extend(featureX[80:100])
                self.X2_testing.extend(featureY[80:100])
                self.classlable_testing.extend([2 for i in range(0,20)])
            else:
                self.X1_testing.extend(featureX[130:150])
                self.X2_testing.extend(featureY[130:150])
                self.classlable_testing.extend([3 for i in range(0,20)])
        elif class1 == 2:
            self.X1_testing = featureX[80:100]
            self.X2_testing = featureY[80:100]
            self.classlable_testing= [2 for i in range(0,20)]
            self.X1_testing.extend(featureX[130:150])
            self.X2_testing.extend(featureY[130:150])
            self.classlable_testing.extend([3 for i in range(0,20)])
    def returnFeature(self,index,rd):
        feature = []
        if index == 1:
            feature = rd.IrisX1
        elif index == 2:
            feature = rd.IrisX2
        elif index == 3:
            feature = rd.IrisX3
        else:
            feature = rd.IrisX4
        return feature
    def plotFeatures(self):
        chosenFeatures = self.chosenFeatures.get()
        feature1 = int(chosenFeatures[1])
        feature2 = int(chosenFeatures[6])
        choosenClasses = self.choosenClasses.get()
        c1 = int(choosenClasses[1])
        c2 = int(choosenClasses[6])
        featureX,featureY = self.initilizeData()
        pi = PlotIris()
        plot = self.plotLine.get()
        if plot == 1:
            lineX = range(int(np.min(featureX) - 2), int(np.max(featureX) + 2))
            lineY = [-(self.b + self.w1 * xi) / self.w2 for xi in lineX]
            pi.plot_(featureX, featureY, 'X'+ str(feature1), 'X'+str(feature2),c1,c2, lineX, lineY)
        else:
            pi.plot(featureX, featureY, 'X'+ str(feature1), 'X'+str(feature2),c1,c2)

    def learning(self):
        self.mainAlgo = BackPropagation()
        num_hidden_layer = self.NumberOfHiddenLayers.get()
        num_neurons_layer = self.NumberOfNeuronsInEachLayer.get()
        eta = float(self.learnRate.get())
        epochsNo = int(self.epochsNo.get())
        bias = self.bias.get()
        activation_function = self.activationFunction.get()
        stopping_criteria = self.stoppingCriteria.get()
        threshold = -1
        if stopping_criteria == 2:
            threshold = float(self.errorTh.get())

        # Initializing Training Features
        self.manageTrainingFeatures()
        # Training The NN
        self.mainAlgo.MainAlgorithm(self.training_features, eta, epochsNo
                               , bias, threshold, stopping_criteria,
                               activation_function, num_hidden_layer,
                               num_neurons_layer)


        #plt.plot(w1)

    def testing(self):

        # self.mainAlgo = BackPropagation()
        num_hidden_layer = self.NumberOfHiddenLayers.get()
        num_neurons_layer = self.NumberOfNeuronsInEachLayer.get()
        bias = self.bias.get()
        activation_function = self.activationFunction.get()

        # Initializing Training Features
        self.manageTrainingFeatures()
        # Training The NN
        Output = self.mainAlgo.MainAlgorithmTesting(self.testing_features, bias,
                               activation_function, num_hidden_layer,
                               num_neurons_layer)

        # computing Confusion Matrix
        Confusion_Matrix = [[0 for x in range(4)] for y in range(4)]
        for i in range(len(Output)):
            Y = self.testing_features["Y"][i]
            Confusion_Matrix[Y][Output[i]]+=1
        print(Confusion_Matrix)

        #Computing OverAllAccurcy
        OverAllAccurcy = 0.0
        sum = 0.0
        for i in range(len(Output)):
            Y = self.testing_features["Y"][i]
            if Y == Output[i]:
                sum += 1
        OverAllAccurcy = sum / len(Output)
        print(OverAllAccurcy)

