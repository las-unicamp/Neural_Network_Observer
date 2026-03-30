from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras
from modules.nnm import NormalizationLayer
from modules.nnm import DenormalizationLayer
from modules.nnm import SingleConnected

class Nnc:
    def __init__(self, nnm, parNnc):

        self.nnm = nnm
        
        self.nncDataSize = parNnc['nncDataSize']
        self.learnRate = parNnc['learnRate']
        self.learnEpochs = parNnc['learnEpochs']
        self.horizonLength = parNnc['horizonLength']
        self.nncLayers = parNnc['nncLayers']
        self.controlSaturation = parNnc['controlSaturation']
        self.controlInputsWeight = parNnc['controlInputsWeight']
        self.trainRatio = parNnc['trainRatio']

        if 'activation' in parNnc.keys():
            self.activation = parNnc['activation']
        else:
            self.activation = 'relu'

        self.nControlInputs = self.nnm.nControlInputs
        self.trained = False
        self.active = True
        self.gradDone = False
        self.gradList = []

        self.buildModel()
        self.lossLog = []
        self.lossLogStartIds = []

        self.ignore = False

    def evaluate(self, states):
        if self.active and self.trained and not self.ignore:
            states = np.array([states])
            return self.model.predict(states)[0]
        else:
            return np.zeros(self.nControlInputs)
            
    def setActive(self, active):
        self.active = active

    def buildModel(self):

        # NNC parameters
        umm = self.controlSaturation
        cLayers = self.nncLayers
        m = self.nnm.nControlInputs
        n = self.nnm.nStates
        nLayers = len(cLayers)
        nChain = self.horizonLength

        # Input, normalization and sparsity layers
        inputs = keras.Input([n])
        self.normalizationLayer = NormalizationLayer(np.zeros(n), np.ones(n))
        self.sparsityLayer = SingleConnected(n)
        u = self.normalizationLayer(inputs)

        u = self.sparsityLayer(u)
        # u = keras.layers.Lambda(lambda x: tf.gather(x, [0], axis=1))(u) # TARSUS
        self.denseLayers = []

        # Hidden layers
        for nUnits in cLayers:
            curDense = keras.layers.Dense(nUnits, activation=self.activation,
                kernel_initializer='random_normal',
                bias_initializer='random_normal')

            u = curDense(u)
            self.denseLayers.append(curDense)
        


        # Output layer using sigmoid for control saturation
        curDense = keras.layers.Dense(m, activation = 'sigmoid')
        u = curDense(u)
        self.denseLayers.append(curDense)

        # Saturated control scaling
        amp = 2*umm
        shift = -umm
        scaleFun = lambda x: x*amp + shift
        self.scaleLayer = keras.layers.Lambda(scaleFun)
        u = self.scaleLayer(u)

        # Controller model
        self.model = keras.Model(inputs, u)

        # Building training closed-loop model 
        nnc = self.model
        nnm = self.nnm.model

        trainInputs = keras.Input([n])
        curX = trainInputs
        closedXs = []

        # Get first control input for the NNM
        curU = nnc(curX)
        closedUs = [curU]

        # First output from NNM
        curUX = keras.layers.Concatenate()([curU, curX])
        closedXs.append(nnm(curUX))

        # Consecutive outputs from the NNM
        for i in range(1, nChain):

            curX = closedXs[i-1]
            curU = nnc(curX)
            closedUs.append(curU)

            curUX = keras.layers.Concatenate()([curU, curX])
            closedXs.append(nnm(curUX))

        # Finish training model
        self.closedLoopModel = keras.Model(trainInputs, closedXs)

        # Store control input tensors
        self.closedUs = tf.stack(closedUs, axis=1)
        self.closedXs = tf.stack(closedXs, axis=1)

    def initializeVariables(self, dataManager):

        self.nnm.setNormalization(dataManager)
        self.nnm.buildLoss()

        self.buildLoss()
        self.nnm.sess.run(tf.global_variables_initializer())
        
        self.initializeNormalizationLayers()
        self.nnm.initializeNormalizationLayers()

    def initializeNormalizationLayers(self):

        meanIn = np.array(self.nnm.statesMean)
        stdIn = np.array(self.nnm.statesStd)
        self.normalizationLayer.setNormalizationValues(meanIn, stdIn)

    def resetWeights(self):
        """Reinitialize all model weights and biases (TF 1.x style)."""
        model = self.model
        sess = tf.compat.v1.keras.backend.get_session()
        for layer in model.layers:
            for attr in ['kernel', 'bias', 'recurrent_kernel']:
                if hasattr(layer, attr):
                    var = getattr(layer, attr)
                    init = getattr(layer, f"{attr}_initializer", None)
                    if init is not None:
                        sess.run(var.assign(init(var.shape, var.dtype)))

    def buildLoss(self):

        labelsShape = [self.nnm.nStates]
        self.labelsXs = tf.placeholder(tf.float32, shape=labelsShape)

        self.lossXs = tf.square((self.closedXs-self.labelsXs)/self.nnm.statesStd)
        self.lossXs = tf.reduce_mean(self.lossXs, axis=-1)
        self.lossXs = tf.reduce_mean(self.lossXs, axis=-1)
        self.lossXs = tf.reduce_mean(self.lossXs, axis=-1)
        
        self.lossUs = tf.square((self.closedUs)/self.nnm.controlInputsStd)
        self.lossUs = tf.reduce_mean(self.lossUs, axis=-1)
        self.lossUs = tf.reduce_mean(self.lossUs, axis=-1)
        self.lossUs = tf.reduce_mean(self.lossUs, axis=-1)

        self.lossTotal = self.lossXs + self.controlInputsWeight*self.lossUs

        lRate = self.learnRate

        trainableVars = []
        for layer in self.denseLayers:
            weights = layer.weights
            for w in weights:
                trainableVars.append(w)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lRate)
        self.optimizer = self.optimizer.minimize(self.lossTotal, var_list=trainableVars)

    def update(self, data):
        self.updateSparsityLayer()
        self.train(data)
        self.trained = True

    def train(self, data):
        nTrain = round(self.nncDataSize*self.trainRatio)
        nTest = self.nncDataSize - nTrain

        trainData, testData = data.getRandomSamples(nTrain, nTest)

        trainDataIn = trainData.statesIn
        testDataIn = testData.statesIn

        reference = self.nnm.linearManager.equilibrium

        printStr = "    Training NNC: {}/{}, statesLoss: {:.4e}, "+\
            "controlInputsLoss: {:.4e}, totalLoss: {:.4e}, testLoss: {:.4e}"

        self.lossLogStartIds.append(len(self.lossLog))
        inputTensor = self.closedLoopModel.input

        
        bestWeights = self.model.get_weights()
        bestLoss = float('inf')
        
        with self.nnm.sess.as_default():
            for i in range(self.learnEpochs):
                
                self.optimizer.run(feed_dict={
                    inputTensor: trainDataIn, 
                    self.labelsXs: reference})

                lossXsVal = self.nnm.sess.run(self.lossXs,feed_dict={
                    inputTensor: trainDataIn,
                    self.labelsXs: reference})

                lossUsVal = self.nnm.sess.run(self.lossUs,feed_dict={
                    inputTensor: trainDataIn,
                    self.labelsXs: reference})

                lossTotalVal = self.nnm.sess.run(self.lossTotal,feed_dict={
                    inputTensor: trainDataIn,
                    self.labelsXs: reference})

                lossTestVal = self.nnm.sess.run(self.lossTotal,feed_dict={
                    inputTensor: testDataIn,
                    self.labelsXs: reference})

                print(printStr.format(
                    i+1, self.learnEpochs, 
                    lossXsVal, lossUsVal, lossTotalVal, lossTestVal), 
                    end='\r')

                self.lossLog.append(
                    [lossXsVal, lossUsVal, lossTotalVal, lossTestVal])
                
                if bestLoss > lossTotalVal:
                    bestWeights = self.model.get_weights()
                    bestLoss = lossTotalVal

            self.model.set_weights(bestWeights)
            
            print(printStr.format(
                i+1, self.learnEpochs, 
                lossXsVal, lossUsVal, lossTotalVal, lossTestVal))

            


    def updateSparsityLayer(self):
        w, _ = self.nnm.sparsityLayer.getSparseMap(opt='states')
        w = np.array(w)
        self.sparsityLayer.set_weights([w])

        

    def plotLoss(self, plt, block=True, f=None):
        import matplotlib.pyplot as plt

        log = np.array(self.lossLog)

        plt.figure(f)
        plt.semilogy(log)
        plt.xlabel('Epoch')
        plt.legend(
            ['lossXsVal', 'lossUsVal',
                'lossTotalVal', 'lossTestVal'])
        for x in self.lossLogStartIds:
            plt.axvline(x=x, color='k', linewidth = 0.5)

        plt.show(block=block)

    def save(self, index, dir):

        label = "nnc_weights"
        saveStr = self.setupString(label, dir, index=index)
        self.model.save_weights(saveStr)
        #print("    Saved: " + saveStr)

        label = "nnc_log"
        saveStr = self.setupString(label, dir, suffix='.npy', log=True)
        np.save(saveStr, self.lossLog)
        #print("    Saved: " + saveStr)

        label = "nnc_log_id"
        saveStr = self.setupString(label, dir, suffix='.npy', log=True)
        np.save(saveStr, self.lossLogStartIds)
        #print("    Saved: " + saveStr)
        
    def load(self, index, dir):

        label = "nnc_weights"
        loadStr = self.setupString(label, dir, index=index)
        self.model.load_weights(loadStr)

        label = "nnc_log"
        loadStr = self.setupString(label, dir, suffix='.npy', log=True)
        self.lossLog = self.loadLog(loadStr)
        
        label = "nnc_log_id"
        loadStr = self.setupString(label, dir, suffix='.npy', log=True)
        self.lossLogStartIds = self.loadLog(loadStr)

        self.clipLogs(index)
        self.trained = True

    def setupString(self, label, dir, index=0, suffix='', log=False):

        if log:
            saveStr = dir + '/' + label + suffix
        else:
            saveStr = dir + '/{:03d}/' + label + suffix
            saveStr = saveStr.format(index)

        return saveStr

    def loadLog(self, path):
        return np.load(path).tolist()

    def clipLogs(self, index):

        if index+1 < len(self.lossLogStartIds):
            clipLossId = self.lossLogStartIds[index+1]
            self.lossLog = self.lossLog[:clipLossId]
            self.lossLogStartIds = self.lossLogStartIds[:index+1]

    def getLinearizedGains(self):
        
        # NNC trained model
        model = self.model
        m = self.nnm.nControlInputs
        n = self.nnm.nStates
        sess = self.nnm.sess

        # Get the relevant positions acording to nonzero values in sparsity layer

        jacFull = []
        printStr = "    Linearizing NNC model ({}/{})"

        # Compute Jacobian
        for i in range(m):
            print(printStr.format(i+1,m), end='\r')

            # Compute gradients
            if not self.gradDone:
                gradFunc = tf.gradients(model.output[:,i], model.input)
                gradFunc = tf.reshape(gradFunc[0],[-1,1,n])

                # Building 'K' matrix
                self.gradList.append(gradFunc)

            gradFunc = self.gradList[i]

            # Append to Jacobians
            jacFull.append(gradFunc)


        # n x ns matrix, which can be used to convert a computed 
        # eigenvector matrix back to the original number of states
        controlMatrix = jacFull
        controlMatrix = tf.concat(controlMatrix,1)

        print('')

        self.gradDone = True

        x0 = self.nnm.linearManager.equilibriumLog[-1][-1]
        return sess.run(controlMatrix, feed_dict = {model.input:[x0]})[0,:,:]