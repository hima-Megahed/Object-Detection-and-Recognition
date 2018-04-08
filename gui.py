import tkinter as tk
from BackPropagation import BackPropagation
import numpy as np


class GUI:
    def __init__(self):
        """ Initializing main components and properties"""
        self.root = tk.Tk()
        self.learnRate = tk.IntVar(self.root)
        self.learnRate.set(0.01)
        self.epochsNo = tk.IntVar(self.root)
        self.epochsNo.set(4)
        self.bias = tk.IntVar(self.root)
        self.bias.set(0)
        self.root.resizable(height=True, width=True)
        self.root.title("Object Detection")
        self.root.geometry("500x500")
        self.errorThreshold = tk.DoubleVar(self.root)
        self.errorThreshold.set(0.001)
        self.NumberOfHiddenLayers = tk.IntVar(self.root)
        self.NumberOfHiddenLayers.set(3)
        self.NumberOfNeuronsInEachLayer = tk.IntVar(self.root)
        self.NumberOfNeuronsInEachLayer.set(5)
        self.activationFunction = tk.IntVar()
        self.activationFunction.set(1)
        self.stoppingCriteria = tk.IntVar()
        self.stoppingCriteria.set(1)
        self.initialize_components()
        self.root.mainloop()

    def initialize_components(self):
        tk.Label(self.root, text="# Of Hidden Layers")\
            .place(relx=0.11, rely=0.05)
        tk.Entry(self.root, width=10, textvariable=self.NumberOfHiddenLayers)\
            .place(relx=0.64, rely=0.05)
        tk.Label(self.root, text="# Of Neurons In Each Layer")\
            .place(relx=0.11, rely=0.14)
        tk.Entry(self.root, width=10,
                 textvariable=self.NumberOfNeuronsInEachLayer)\
            .place(relx=0.64, rely=0.14)
        tk.Label(self.root, text="Enter Learning Rate(eta):")\
            .place(relx=0.11, rely=0.23)
        tk.Entry(self.root, width=10, textvariable=self.learnRate)\
            .place(relx=0.64, rely=0.23)
        tk.Label(self.root, text="Enter Number of Epochs:")\
            .place(relx=0.11, rely=0.32)
        tk.Entry(self.root, width=10, textvariable=self.epochsNo)\
            .place(relx=0.64, rely=0.32)
        tk.Label(self.root, text="Choose an activation function")\
            .place(relx=0.03, rely=0.41)
        tk.Radiobutton(self.root,
                       text="Sigmoid",
                       variable=self.activationFunction,
                       value=1).place(relx=0.03, rely=0.45)
        tk.Radiobutton(self.root,
                       text="Hyperbolic Tangent Sigmoid",
                       variable=self.activationFunction,
                       value=2).place(relx=0.03, rely=0.49)
        tk.Label(self.root, text="Bias:").place(relx=0.70, rely=0.41)
        tk.Checkbutton(self.root, variable=self.bias)\
            .place(relx=0.78, rely=0.41)
        tk.Label(self.root, text="Choose The Stopping Criteria")\
            .place(relx=0.03, rely=0.58)
        tk.Radiobutton(self.root,
                       text="# of Epochs",
                       variable=self.stoppingCriteria,
                       value=1).place(relx=0.03, rely=0.62)
        tk.Radiobutton(self.root,
                       text="MSE Threshold",
                       variable=self.stoppingCriteria,
                       value=2).place(relx=0.03, rely=0.66)

        tk.Entry(self.root, width=10, textvariable=self.errorThreshold)\
            .place(relx=0.64, rely=0.66)

        tk.Button(self.root, text="Plotting", width=10, fg="Black",
                  bg="light Gray", command=lambda: self.plotFeatures())\
            .place(relx=0.03, rely=0.80)
        tk.Button(self.root, text="Learning", width=10, fg="Black",
                  bg="light Gray", command=lambda: self.learning())\
            .place(relx=0.35, rely=0.80)
        tk.Button(self.root, text="Testing", width=10, fg="Black",
                  bg="light Gray", command=lambda: self.testing())\
            .place(relx=0.67, rely=0.80)
