#!/user/bin/python3
import json
import tkinter as tk
from tkinter import ttk

import pickle

from BackPropagation import BackPropagation
from DataManipulation import TrainingData, TestingData
from RBF import RadialBasisFunction

from segmentation import SegmentationEngine
from Highlighting import Highlight
from N_PCA import N_PCA


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
        self.errorThreshold.set(0.0004)
        self.errorThresholdRBF = tk.DoubleVar(self.root)
        self.errorThresholdRBF.set(0.092)
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
        self.RBFweights = self.RBFcentroids = None
        self.RBF = None
        training_tmp = TrainingData()
        self.TrainingData = TrainingData.read(training_tmp)
        self.PCA_TFeatures = TrainingData.apply_pca(training_tmp)

        self.back_propagation = BackPropagation()

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
        tk.Entry(page2, width=17, textvariable=self.errorThresholdRBF) \
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
        tk.Button(self.root, text="Draw PCA Graph", width=13, fg="Black",
                  bg="light Gray", command=lambda: self.pca_graph())\
            .place(relx=0.25, rely=0.84)
        tk.Button(self.root, text="Testing RealTime", width=13, fg="Black",
                  bg="light Gray", command=lambda: self.test_Run())\
            .place(relx=0.48, rely=0.84)
        tk.Button(self.root, text="Testing Fixed", width=10, fg="Black",
                  bg="light Gray", command=lambda: self.fixed_test()) \
            .place(relx=0.70, rely=0.84)


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

            self.RBF = RadialBasisFunction(self.PCA_TFeatures,self.learnRate.get(),
                                      self.epochsNo.get(),self.errorThresholdRBF.get(),
                                      self.NumberOfNeuronsRBF.get())

            self.RBFweights, self.RBFcentroids = self.RBF.mseTrain()
            print(self.RBFweights)
            print(self.RBFcentroids)
            it = 0


    def test_Run(self):
        seg = SegmentationEngine()
        TestData = seg.get_tests()
        testing = TestingData()
        testing_features = list()

        [testing_features.extend(img['ObjectImgsGrey']) for img in TestData]
        PCA_Test_Features = testing.read(testing_features)

        if self.tab_control.index(self.tab_control.select()) == 0:
                res = self.back_propagation.test_literal(PCA_Test_Features,
                                                         self.bias.get(),
                                                         self.activationFunction.get())
        else:
            '''
            self.RBFweights = [[0.45371985821449845, 0.06811024565477831, 0.2940796547648568, 0.6606915411151577, 0.9950180045224093, 0.02550053502215005, 0.47630276025957313, 0.5666971478790688, 0.6840765849138305, 0.8951444851538983],
           [0.35869122328654335, 0.10741503134446385, 0.5572717927090857, -0.04623951368317306, 0.5954774379337506, 0.5619789012789034, 0.5427785394604984, 0.5101290224897486, 0.9995867009733511, 0.7453457949468193],
           [0.44407010761406623, 0.07281987421068473, 0.5672012476704367, 0.3553263366406079, 0.1502513579979443, 0.5182353912876106, 0.36262066087711986, 0.394031687120829, 0.023788731307406476, 1.047162022998752],
           [0.4075411649755102, 0.3137285050098999, 0.7097310213635801, 0.34776487945275864, 0.8781420604271841, 0.33047094902334634, 0.4557338594425149, 0.06923173823499171, 1.0348967044766273, 0.7166868842393158],
           [0.5266955638698242, 1.0040749918632916, 0.6350305714595198, 0.4283644850709838, 0.8258594509763931, 0.5175075258045693, 0.1286058524622251, 0.48302697212859885, 0.8062942128095698, 0.41393814589378675]]
            self.RBFcentroids = [[21.86460817567456, -0.7667202982276109, -1.0030206582982089, -13.252661953989792, -9.750527721330926, -8.659212015515601, 1.9818268992033294, -0.41579657674192133, 0.09966031569823919, -7.584643581630902, -3.1103562907336912, 1.2033736975299079, -4.396636555815659, 5.312587854742299, -5.250983817196718, 0.47878287252825324, -0.43399367869289646, -5.132211159816143, -0.8031082132792703, 0.9651533773587512, -2.6894306654770843, 1.0211058523909473, 3.11085487578819, 0.3152442773199197],
             [30.927843602305288, -30.764362386106786, -0.19197723214502696, 7.81906491781681, -0.15171685056899387, 2.866381143770637, -3.7674352397574946, -0.15766425621605648, 3.647750946580976, 8.948092819277496, 2.4323826701947784, 3.610615423720739, 1.7870601185187096, -0.11971999371987074, -1.7699274289934468, 1.2399629368063636, -0.9589909113324216, 1.6030154979651605, -0.36201921314392066, -0.23860552467718973, 2.520645159977222, -0.5022923737137814, -1.3582436467731034, -1.2739584022327402],
             [-28.101275967174654, -6.102729002047614, -10.435385023834439, -0.4692510462616373, 5.250334259522102, -2.4967572700761527, -7.0634710024234, -7.971447605948123, -2.0852620102554633, 0.12742405476766672, -1.8862295317648505, -2.059457913855882, -0.23312377240701032, 5.347598982274169, -2.0873303311519673, -0.08073228456590131, -0.06649112208859176, 2.7541245377017374, 0.4000357447438191, -0.6366121433879353, 1.79837210211556, 2.6101515261490436, 1.002475571995773, -1.128368690211698],
             [-3.2633933953624803, -0.4766092948782351, 0.8096006995507494, -0.5590412909520593, 5.237863386709334, -11.637327675140165, 3.627226152301349, 7.939226428079072, 0.08452945143417656, 0.2586383566666832, -0.43293624951289833, -2.372560234109779, 1.9873414288916549, -0.9063254249250341, 2.7957883843991533, -5.019386789920255, -2.2746260553738686, 0.811913112660795, 4.340083320714989, 0.09194480258219051, 1.519338025776345, -2.3243137156335036, -0.937820693688231, 3.1567936190569608],
             [10.449462554279267, 8.945168795070224, 1.4576171083854341, -15.806264891114731, 9.291727318171526, 13.369812722810556, -2.21298941552385, 1.9699771236494321, 1.6255430325673235, -3.540111455128917, -6.0504064885440085, -2.787501234538542, 2.065465946902695, -0.706367826954652, 5.639993288941454, 3.758146668060485, 1.9312222240436163, 3.6760531809985637, -1.4462373277526375, -1.6641231865158026, -3.245756267175688, -1.0498749024239042, -2.7504631377552484, -0.6444294843064547],
             [10.32846521717222, 16.28555267142517, -0.6997350106914053, 17.31972275226288, 34.977751232933805, -14.519636638639703, 11.536546981542019, -6.191779977467325, -4.770986798254331, -0.28072456472684815, 4.591047607877742, 8.28688370563612, -2.9906443519453174, -13.784063853803872, -2.9626027783899556, -0.5249492898540301, 4.6879810156731985, -2.4010188542052413, -6.107462211754136, -4.972689295146779, -5.261979962089118, 5.103042727494589, 1.653820328680621, -0.5834786083855864],
             [-6.635137921211104, -2.6992453077583107, -25.21246403118692, 22.04830254516172, -20.179012639399595, 9.338595208606332, 19.62354906116619, -4.656093905680765, -6.659282648102137, 4.092283865058338, -0.6053427095201943, 2.8759499915970563, -6.3188492067952975, -1.4428650963182768, 10.793456434821097, 3.5046651159275632, -1.1256597399532677, 4.747461379413141, 4.889832203498173, 4.6396966381636195, -9.22120063031249, -2.3816603848267253, 4.749909657289091, -0.2541493482121517],
             [-6.957031137244706, 16.54098855763076, -15.051764970812284, 35.729540316048194, -6.262319552205901, 15.325936885664355, -14.489282952277575, 13.569697649351403, 19.03963715177163, -16.28355755534008, 0.018698798076038525, -3.814741783245267, 2.850823361544021, -4.001058924462071, -4.057072315753233, 1.3443867078959701, -0.19502133961964938, -5.434512757663278, -1.193049782368929, -1.0172843099505455, 1.092320306671487, 0.33928819505284713, 1.0038195957127107, 0.9433478508169476],
             [0.8014183217275956, 32.78198651353656, 7.411380767466963, -2.6568680861461087, -10.779323364809796, 7.126607543030253, -1.5351665904435698, -3.6568333158863733, -5.352581894837452, 8.636110569170318, 3.664762627228216, 2.2713989517495095, 0.8206386233987869, -2.369733521378764, -0.052106029922499054, 0.2130341895419979, -1.351972401344895, -2.366567616433493, -0.5534961103879534, -0.40477194733662203, 4.968630703856928, 0.3513705711225432, 0.7300538524406791, 0.31587939932440023],
             [-31.70380833526402, -17.199080303205733, 28.222550676798953, 0.35289688784176665, -8.159843885319791, 1.1403380714734066, 6.83772967481114, 2.094006010428835, 0.16622266492139737, -4.225298222896413, 7.231555937120806, -1.2568399187550172, -0.6938924138777905, -2.443702946439689, -1.1542707960765106, -0.4967031184180817, 4.219186537864674, -2.258475038568508, -3.527606938759501, 3.777994086834625, -2.2711235096908045, -1.8324746109551562, -3.4313874630070718, -2.1788134899578253]]
            '''
            res = self.RBF.run_test(PCA_Test_Features,
                                   self.RBFweights,
                                   self.RBFcentroids)

        res_ind = 0
        for img in TestData:
            highlight = Highlight()
            highlight.HighlightObjects(img, res[res_ind: res_ind + len(img['ObjectImgsGrey'])])
            res_ind += len(img['ObjectImgsGrey'])

        a = 0
        '''when run the app call run func'''
        '''res = self.RBF.run(self.PCA_Test_Features[0], self.RBFweights,
                           self.RBFcentroids)
        print(res)'''

    def fixed_test(self):
        test_data = TestingData()
        PCA_Test_Features = test_data.read_test_data_run()
        if self.tab_control.index(self.tab_control.select()) == 0:
                res = self.back_propagation.test(PCA_Test_Features,
                                                 self.bias.get(),
                                                 self.activationFunction.get())
        else:
            res = self.RBF.mseTest(PCA_Test_Features,
                                   self.RBFweights,
                                   self.RBFcentroids)
        print(res)

        '''when run the app call run func'''
        res = self.RBF.run(self.PCA_Test_Features[0],
                           self.RBFweights,
                           self.RBFcentroids)
        print(res)

        print("test for commit")



    @staticmethod
    def pca_graph():
        training_tmp = TrainingData()
        TrainingData.read(training_tmp)
        TrainingData.graph_pcs(training_tmp)

    def get_neural_PCA(self):
        training_tmp = TrainingData()
        self.PCA_TFeatures = training_tmp.train_Neural_pca(self.TrainingData)
        self.PCA_TFeatures = training_tmp.apply_Neural_pca(self.TrainingData)

