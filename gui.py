#!/user/bin/python3
import tkinter as tk
from tkinter import ttk
from BackPropagation import BackPropagation
from DataManipulation import TrainingData, TestingData
import numpy as np
from RBF import RadialBasisFunction


class GUI:
    def __init__(self):
        """ Initializing main components and properties"""
        self.root = tk.Tk()
        self.learnRate = tk.DoubleVar(self.root)
        self.learnRate.set(0.3)
        self.epochsNo = tk.IntVar(self.root)
        self.epochsNo.set(2000)
        self.bias = tk.IntVar(self.root)
        self.bias.set(1)
        self.root.resizable(height=False, width=False)
        self.root.title("Object Detection")
        self.root.geometry("500x500")
        self.errorThreshold = tk.DoubleVar(self.root)
        self.errorThreshold.set(0.001)
        self.NumberOfHiddenLayers = tk.IntVar(self.root)
        self.NumberOfHiddenLayers.set(2)
        self.NumberOfNeuronsInEachLayer = tk.IntVar(self.root)
        self.NumberOfNeuronsInEachLayer.set(6)
        self.activationFunction = tk.IntVar()
        self.activationFunction.set(1)
        self.stoppingCriteria = tk.IntVar()
        self.stoppingCriteria.set(1)
        self.NumberOfNeuronsRBF = tk.IntVar()
        self.NumberOfNeuronsRBF.set(4)
        self.tab_control = ttk.Notebook(self.root)

        training_tmp = TrainingData()
        self.TrainingData = TrainingData.read(training_tmp)
        self.PCA_TFeatures = TrainingData.apply_pca(training_tmp)
        self.back_propagation = None

        testing_tmp = TestingData()
        TestingData.read(testing_tmp)
        self.PCA_Test_Features = TestingData.apply_pca(testing_tmp)

        self.initialize_components()
        self.root.mainloop()

    def initialize_components(self):
        """ Setting Layout Components for GUI """

        # Setting Tab Controls
        self.tab_control.grid(row=1, column=0, columnspan=100, rowspan=100,
                         sticky='NSWE')

        page1 = ttk.Frame(self.tab_control)
        page2 = ttk.Frame(self.tab_control)
        self.tab_control.add(page1, text='MLP')
        self.tab_control.add(page2, text='RBF')
        self.tab_control.pack(expand=1, fill='both')

        # Setting Items of Tab1 MLP
        tk.Label(page1, text="# Of Hidden Layers").place(relx=0.03, rely=0.05)
        tk.Entry(page1, width=17, textvariable=self.NumberOfHiddenLayers)\
            .place(relx=0.64, rely=0.05)
        tk.Label(page1, text="# Of Neurons In Each Layer")\
            .place(relx=0.03, rely=0.14)
        tk.Entry(page1, width=17,
                 textvariable=self.NumberOfNeuronsInEachLayer)\
            .place(relx=0.64, rely=0.14)
        tk.Label(page1, text="Enter Learning Rate(eta):")\
            .place(relx=0.03, rely=0.23)
        tk.Entry(page1, width=17, textvariable=self.learnRate)\
            .place(relx=0.64, rely=0.23)
        tk.Label(page1, text="Enter Number of Epochs:")\
            .place(relx=0.03, rely=0.32)
        tk.Entry(page1, width=17, textvariable=self.epochsNo)\
            .place(relx=0.64, rely=0.32)
        tk.Label(page1, text="Choose an activation function")\
            .place(relx=0.03, rely=0.41)
        tk.Radiobutton(page1,
                       text="Sigmoid",
                       variable=self.activationFunction,
                       value=1).place(relx=0.03, rely=0.45)
        tk.Radiobutton(page1,
                       text="Hyperbolic Tangent Sigmoid",
                       variable=self.activationFunction,
                       value=2).place(relx=0.03, rely=0.50)
        tk.Label(page1, text="Bias:").place(relx=0.70, rely=0.41)
        tk.Checkbutton(page1, variable=self.bias)\
            .place(relx=0.78, rely=0.41)
        tk.Label(page1, text="Choose The Stopping Criteria")\
            .place(relx=0.03, rely=0.58)
        tk.Radiobutton(page1,
                       text="# of Epochs",
                       variable=self.stoppingCriteria,
                       value=1).place(relx=0.03, rely=0.62)
        tk.Radiobutton(page1,
                       text="MSE Threshold",
                       variable=self.stoppingCriteria,
                       value=2).place(relx=0.03, rely=0.67)
        tk.Radiobutton(page1,
                       text="Cross Validation",
                       variable=self.stoppingCriteria,
                       value=3).place(relx=0.03, rely=0.72)

        tk.Entry(page1, width=17, textvariable=self.errorThreshold)\
            .place(relx=0.64, rely=0.67)
        # ================================================================== #
        # Setting Items of Tab2 RBF
        tk.Label(page2, text="Number of hidden neurons:")\
            .place(relx=0.03, rely=0.1)
        tk.Entry(page2, width=17, textvariable=self.NumberOfNeuronsRBF)\
            .place(relx=0.64, rely=0.1)
        tk.Label(page2, text="MSE Threshold:") \
            .place(relx=0.03, rely=0.2)
        tk.Entry(page2, width=17, textvariable=self.errorThreshold) \
            .place(relx=0.64, rely=0.2)
        tk.Label(page2, text="Enter Learning Rate(eta):") \
            .place(relx=0.03, rely=0.3)
        tk.Entry(page2, width=17, textvariable=self.learnRate) \
            .place(relx=0.64, rely=0.3)
        tk.Label(page2, text="Enter Number of Epochs:") \
            .place(relx=0.03, rely=0.4)
        tk.Entry(page2, width=17, textvariable=self.epochsNo) \
            .place(relx=0.64, rely=0.4)
        # ================================================================== #
        tk.Button(self.root, text="Train Model", width=10, fg="Black",
                  bg="light Gray", command=lambda: self.train_model())\
            .place(relx=0.03, rely=0.84)
        tk.Button(self.root, text="Init Network", width=10, fg="Black",
                  bg="light Gray", command=lambda: self.init())\
            .place(relx=0.27, rely=0.84)
        tk.Button(self.root, text="Testing", width=10, fg="Black",
                  bg="light Gray", command=lambda: self.test())\
            .place(relx=0.51, rely=0.84)

    def train_model(self):
        # Multi Layer Perceptron
        if self.tab_control.index(self.tab_control.select()) == 0:
            num_hidden_layer = self.NumberOfHiddenLayers.get()
            num_neurons_layer = self.NumberOfNeuronsInEachLayer.get()
            learn_rate = self.learnRate.get()
            epoch_number = int(self.epochsNo.get())
            bias = self.bias.get()
            activation_function = self.activationFunction.get()
            stopping_criteria = self.stoppingCriteria.get()
            threshold = -1
            if stopping_criteria == 2:
                threshold = self.errorThreshold.get()

            # ٍ Shuffling Input Data
            self.PCA_TFeatures = self.back_propagation.shuffle_data(
                self.PCA_TFeatures)
            # Training The NN
            self.back_propagation.main_algorithm(self.PCA_TFeatures, learn_rate,
                                            epoch_number, bias, threshold,
                                            stopping_criteria,
                                            activation_function,
                                            num_hidden_layer,
                                            num_neurons_layer,
                                            25)
        # Radial Basis Function
        else:
            RBF = RadialBasisFunction(self.PCA_TFeatures,self.learnRate.get(),
                                      self.epochsNo.get(),self.errorThreshold.get(),
                                      self.NumberOfNeuronsRBF.get())
            weights = RBF.mseTrain() #weights[ [numHiddenNeurons] ] -> outter list size equals number of output neurons, inner  equals num HiddenNeurons

            RBF.mseTest(self.PCA_Test_Features, weights)      # this calling for testing

        return 0

    def init(self):
        self.back_propagation = BackPropagation(
            self.NumberOfNeuronsInEachLayer.get(),
            self.NumberOfHiddenLayers.get())

    def test(self):
        res = self.back_propagation.test(self.PCA_Test_Features,
                                         self.bias.get(),
                                         self.activationFunction.get())
        return 0