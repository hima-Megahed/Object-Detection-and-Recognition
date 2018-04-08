#!/user/bin/python3
import tkinter as tk
from tkinter import ttk
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
        self.root.resizable(height=False, width=False)
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
        self.NumberOfNeuronsRBF = tk.IntVar()
        self.NumberOfNeuronsRBF.set(4)
        self.initialize_components()
        self.root.mainloop()

    def initialize_components(self):
        """ Setting Layout Components for GUI """

        # Setting Tab Controls
        tab_control = ttk.Notebook(self.root)
        tab_control.grid(row=1, column=0, columnspan=100, rowspan=100,
                         sticky='NSWE')

        page1 = ttk.Frame(tab_control)
        page2 = ttk.Frame(tab_control)
        tab_control.add(page1, text='MLP')
        tab_control.add(page2, text='RBF')
        tab_control.pack(expand=1, fill='both')

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
        tk.Button(self.root, text="Plotting", width=10, fg="Black",
                  bg="light Gray", command=lambda: self.plotFeatures())\
            .place(relx=0.03, rely=0.80)
        tk.Button(self.root, text="Learning", width=10, fg="Black",
                  bg="light Gray", command=lambda: self.learning())\
            .place(relx=0.35, rely=0.80)
        tk.Button(self.root, text="Testing", width=10, fg="Black",
                  bg="light Gray", command=lambda: self.testing())\
            .place(relx=0.67, rely=0.80)
