from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras
from modules.linearManager import LinearManager

class Nnm:
    def __init__(self, parNnm, parEquilibrium):

        # NNM parameters
        self.nnmDataSize = parNnm['nnmDataSize']
        self.learnRate = parNnm['learnRate']
        self.learnEpochs = parNnm['learnEpochs']
        self.nnmLayers = parNnm['nnmLayers']
        self.trainRatio = parNnm['trainRatio']
        self.hiddenLayersL2Reg = parNnm['hiddenLayersL2Reg']
        self.useSparsityLayer = parNnm['useSparsityLayer']
        self.sparsityLayerL1Reg = parNnm['sparsityLayerL1Reg']
        self.sparsityTolerance = parNnm['sparsityTolerance']
        self.nControlInputs = parNnm['nControlInputs']
        self.nStates = parNnm['nStates']

        # Equilibrium parameters
        self.initialGuess = parEquilibrium['initialGuess']
        self.nNewtonSteps = parEquilibrium['nNewtonSteps']

        self.buildModel()
        self.lossLog = []
        self.lossLogStartIds = []

        self.linearManager = LinearManager(self.initialGuess, self.nNewtonSteps)

        self.trainable = True
        self.sess = tf.compat.v1.keras.backend.get_session()
                
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

    def buildModel(self):

        m = self.nControlInputs
        n = self.nStates
        inputs = keras.Input([m+n])
        self.sparsityLayer = SingleConnected(inputDim=m+n, trainable=self.useSparsityLayer, 
            sparseTol=self.sparsityTolerance, nControlInputs=self.nControlInputs)

        self.normalizationLayer = NormalizationLayer(np.zeros(m+n),np.ones(m+n))
        self.denormalizationLayer = DenormalizationLayer(np.zeros(n),np.ones(n))

        x = self.normalizationLayer(inputs)
        x = self.sparsityLayer(x)
        self.denseLayers = []

        for nUnits in self.nnmLayers:
            curDense = keras.layers.Dense(nUnits, activation='relu',
                kernel_initializer='random_normal',
                bias_initializer='random_normal')
            self.denseLayers.append(curDense)
            x = curDense(x)

        curDense  = keras.layers.Dense(n, activation='linear',
            kernel_initializer='random_normal',
            bias_initializer='random_normal')
        self.denseLayers.append(curDense)
        x = curDense(x)
        outputs = self.denormalizationLayer(x)

        self.model = keras.Model(inputs, outputs)


    def update(self, dataManager):
        data = dataManager.gatherData()
        self.train(data)
        self.sparsityLayer.truncateWeights()

        self.linearManager.update(self, dataManager)

    def train(self,data):

        nTrain = round(self.nnmDataSize*self.trainRatio)
        nTest = self.nnmDataSize - nTrain
        trainData, testData = data.getRandomSamples(nTrain, nTest)

        trainDataIn = (trainData.controlInputs, trainData.statesIn)
        trainDataIn = np.concatenate(trainDataIn,axis=1)
        trainDataOut = trainData.statesOut

        testDataIn = (testData.controlInputs, testData.statesIn)
        testDataIn = np.concatenate(testDataIn,axis=1)
        testDataOut = testData.statesOut

        printStr = "    Training NNM: {}/{}, errLoss: {:.4e}, reg1Loss: {:.4e}, "+\
            "reg2Loss: {:.4e}, totalLoss: {:.4e}, testLoss: {:.4e}, nSensors: {:d}   "

        self.lossLogStartIds.append(len(self.lossLog))
        bestWeights = self.model.get_weights()
        bestLoss = float('inf')

        with self.sess.as_default():
            for i in range(self.learnEpochs):

                self.optimizer.run(feed_dict={
                    self.model.input: trainDataIn, 
                    self.labels: trainDataOut})

                lossErrVal = self.sess.run(self.lossErr,feed_dict={
                    self.model.input: trainDataIn,
                    self.labels: trainDataOut})

                lossRegL1Val = self.sess.run(self.lossRegL1,feed_dict={
                    self.model.input: trainDataIn,
                    self.labels: trainDataOut})

                lossRegL2Val = self.sess.run(self.lossRegL2,feed_dict={
                    self.model.input: trainDataIn,
                    self.labels: trainDataOut})

                lossTotalVal = self.sess.run(self.lossTotal,feed_dict={
                    self.model.input: trainDataIn,
                    self.labels: trainDataOut})

                lossTestVal = self.sess.run(self.lossTotal,feed_dict={
                    self.model.input: testDataIn,
                    self.labels: testDataOut})
                
                nSensors = self.sparsityLayer.getNSensors()

                print(printStr.format(
                    i+1, self.learnEpochs, 
                    lossErrVal, lossRegL1Val, lossRegL2Val, lossTotalVal, lossTestVal, nSensors), 
                    end='\r')

                self.lossLog.append(
                    [lossErrVal, lossRegL1Val, lossRegL2Val, lossTotalVal, lossTestVal])

                if bestLoss > lossTotalVal:
                    bestWeights = self.model.get_weights()
                    bestLoss = lossTotalVal

        self.model.set_weights(bestWeights)

        
        print(printStr.format(
            i+1, self.learnEpochs, 
            lossErrVal, lossRegL1Val, lossRegL2Val, lossTotalVal, lossTestVal, nSensors))
    

    def buildLoss(self):

        m = self.nControlInputs
        n = self.nStates
        modelOut = self.model.output
        self.labels = tf.placeholder(tf.float32, shape=modelOut.shape)

        self.lossErr = tf.square((self.labels-modelOut)/self.statesStd)
        self.lossErr = tf.reduce_mean(self.lossErr, axis=-1)
        self.lossErr = tf.reduce_mean(self.lossErr, axis=-1)

        mask = np.concatenate([np.zeros(m), np.ones(n)])
        self.lossRegL1 = tf.abs(self.sparsityLayer.w)*mask
        self.lossRegL1 = tf.reduce_sum(self.lossRegL1, axis=-1)

        self.lossRegL2 = 0
        for curDense in self.denseLayers:
            curWeights = curDense.weights[0]
            self.lossRegL2 = self.lossRegL2 + tf.reduce_sum(tf.square(curWeights))

        l1Reg = self.sparsityLayerL1Reg
        l2Reg = self.hiddenLayersL2Reg
        self.lossTotal = self.lossErr + l1Reg*self.lossRegL1 + l2Reg*self.lossRegL2

        lRate = self.learnRate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lRate)
        self.optimizer = self.optimizer.minimize(self.lossTotal)

    def setNormalization(self, dataManager):

        self.controlInputsMean = dataManager.controlInputsMean
        self.controlInputsStd = dataManager.controlInputsStd
        self.statesMean = dataManager.statesMean
        self.statesStd = dataManager.statesStd


    def initializeNormalizationLayers(self):

        meanIn = np.concatenate((self.controlInputsMean, self.statesMean))
        stdIn = np.concatenate((self.controlInputsStd, self.statesStd)) 
        self.normalizationLayer.setNormalizationValues(meanIn, stdIn)

        meanOut = np.array(self.statesMean)
        stdOut = np.array(self.statesStd)
        self.denormalizationLayer.setNormalizationValues(meanOut, stdOut)


    def save(self, index, dir):

        label = "nnm_weights"
        saveStr = self.setupString(label, dir, index=index)
        self.model.save_weights(saveStr)
        #print("    Saved: " + saveStr)

        label = "nnm_log"
        saveStr = self.setupString(label, dir, suffix='.npy', log=True)
        np.save(saveStr, self.lossLog)
        #print("    Saved: " + saveStr)

        label = "nnm_log_id"
        saveStr = self.setupString(label, dir, suffix='.npy', log=True)
        np.save(saveStr, self.lossLogStartIds)
        #print("    Saved: " + saveStr)

        label = "nnm_log_equilibrium"
        saveStr = self.setupString(label, dir, suffix='.npy', log=True)
        np.save(saveStr, self.linearManager.equilibriumLog[1:])
        #print("    Saved: " + saveStr)

        label = "nnm_log_sparsity"
        saveStr = self.setupString(label, dir, suffix='.npy', log=True)
        np.save(saveStr, self.linearManager.sparsityLog)
        #print("    Saved: " + saveStr)

        label = "nnm_log_eigVal"
        saveStr = self.setupString(label, dir, suffix='.npy', log=True)
        np.save(saveStr, self.linearManager.eigValLog)
        #print("    Saved: " + saveStr)

        label = "nnm_log_eqFix"
        saveStr = self.setupString(label, dir, suffix='.npy', log=True)
        np.save(saveStr, self.linearManager.eqFixLog)
        #print("    Saved: " + saveStr)

        label = "nnm_eigVec"
        saveStr = self.setupString(label, dir, index=index, suffix='.txt')
        np.savetxt(saveStr, self.linearManager.eigVec)
        #print("    Saved: " + saveStr)

        label = "nnm_eigVal"
        saveStr = self.setupString(label, dir, index=index, suffix='.txt')
        np.savetxt(saveStr, self.linearManager.eigVal)
        #print("    Saved: " + saveStr)

        label = "nnm_mask"
        saveStr = self.setupString(label, dir, index=index, suffix='.txt')
        mask, _ = self.sparsityLayer.getSparseMap()
        np.savetxt(saveStr, mask)
        #print("    Saved: " + saveStr)

    def load(self, index, dir):

        label = "nnm_weights"
        loadStr = self.setupString(label, dir, index=index)
        self.model.load_weights(loadStr)

        label = "nnm_log"
        loadStr = self.setupString(label, dir, suffix='.npy', log=True)
        self.lossLog = self.loadLog(loadStr)
        
        label = "nnm_log_id"
        loadStr = self.setupString(label, dir, suffix='.npy', log=True)
        self.lossLogStartIds = self.loadLog(loadStr)

        label = "nnm_log_equilibrium"
        loadStr = self.setupString(label, dir, suffix='.npy', log=True)
        eqLog = self.loadLog(loadStr)
        for eq in eqLog:
            self.linearManager.equilibriumLog.append(eq)
    
        label = "nnm_log_sparsity"
        loadStr = self.setupString(label, dir, suffix='.npy', log=True)
        self.linearManager.sparsityLog = self.loadLog(loadStr, allowPickle=True)
    
        label = "nnm_log_eigVal"
        loadStr = self.setupString(label, dir, suffix='.npy', log=True)
        self.linearManager.eigValLog = self.loadLog(loadStr, allowPickle=True)
    
        label = "nnm_log_eqFix"
        loadStr = self.setupString(label, dir, suffix='.npy', log=True)
        self.linearManager.eqFixLog = self.loadLog(loadStr)

        self.clipLogs(index)


    def setupString(self, label, dir, index=0, suffix='', log=False):

        if log:
            saveStr = dir + '/' + label + suffix
        else:
            saveStr = dir + '/{:03d}/' + label + suffix
            saveStr = saveStr.format(index)

        return saveStr

    def loadLog(self, path, allowPickle = False):
        return np.load(path, allow_pickle=allowPickle).tolist()

    def plotLoss(self, plt, block=True, f=None):

        log = np.array(self.lossLog)

        plt.figure(f)
        plt.semilogy(log)
        plt.xlabel('Epoch')
        plt.legend(
            ['lossErrVal', 'lossRegL1Val', 'lossRegL2Val',
                'lossTotalVal', 'lossTestVal'])
        for x in self.lossLogStartIds:
            plt.axvline(x=x, color='k', linewidth = 0.5)

        plt.show(block=block)

    def clipLogs(self, index):

        if index+1 < len(self.lossLogStartIds):
            self.linearManager.equilibriumLog = self.linearManager.equilibriumLog[:index+2]
            clipLossId = self.lossLogStartIds[index+1]
            self.lossLog = self.lossLog[:clipLossId]
            self.lossLogStartIds = self.lossLogStartIds[:index+1]
            self.linearManager.sparsityLog = self.linearManager.sparsityLog[:index+1]
            self.linearManager.eigValLog = self.linearManager.eigValLog[:index+1]
            self.linearManager.eqFixLog = self.linearManager.eqFixLog[:index+1]


class SingleConnected(keras.layers.Layer):

    def __init__(self, inputDim=32, trainable=False, sparseTol=1e-3, nControlInputs=0):
        super(SingleConnected, self).__init__()
        wInit = keras.initializers.ones()
        nW = int(inputDim)
        self.inputDim = inputDim
        self.trainable = trainable
        self.sparseTol = sparseTol
        self.nControlInputs = nControlInputs
        self.w = tf.Variable(
            initial_value = wInit(shape=nW, dtype="float32"),
            trainable = trainable
        )
        return

    def get_config(self):
        return {
            'inputDim': self.inputDim, 
            'trainable': self.trainable,
            'sparseTol': self.sparseTol,
            'nControlInputs': self.nControlInputs
            }

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        w = self.w
        return inputs*w

    def truncateWeights(self):
        w = self.get_weights()[0]
        tol = self.sparseTol
        wTruncated = [0.0 if np.abs(wi) < tol else wi for wi in w]
        wTruncated = np.array(wTruncated) 
        self.set_weights([wTruncated])

    def getSparseMap(self, opt='full'):

        if opt == 'full':
            start = 0
        elif opt == 'states':
            start = self.nControlInputs
        
        w = self.get_weights()[0]
        w = w[start:]
        tol = self.sparseTol
        mask = [0.0 if np.abs(wi) < tol else 1.0 for wi in w]

        ids = []
        
        for i in range(w.shape[0]):
            if mask[i] == 1:
                ids.append(i)

        return mask, ids

    def getNSensors(self):
        mask, _ = self.getSparseMap(opt='states')
        return int(np.sum(mask))



class NormalizationLayer(keras.layers.Layer):
    def __init__(self, mean=0.0, std=1.0):
        super(NormalizationLayer, self).__init__()  

        mean = tf.cast(mean, tf.float32)
        std = tf.cast(std, tf.float32)

        self.trainable = False

        self.mean = tf.Variable(
            initial_value = mean,
            trainable = False
        )
        self.std = tf.Variable(
            initial_value = std,
            trainable = False
        )
        
    def get_config(self):
        return {'mean': self.mean, 'std': self.std}
        
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        return (inputs-self.mean)/self.std

    def setNormalizationValues(self, mean, std):
        self.mean.assign(mean)
        self.std.assign(std)
        self.set_weights(np.array([mean, std]))
        
class DenormalizationLayer(keras.layers.Layer):
    def __init__(self, mean=0.0, std=1.0):
        super(DenormalizationLayer, self).__init__()  

        mean = tf.cast(mean, tf.float32)
        std = tf.cast(std, tf.float32)

        self.trainable = False

        self.mean = tf.Variable(
            initial_value = mean,
            trainable = False
        )
        self.std = tf.Variable(
            initial_value = std,
            trainable = False
        )
        
    def get_config(self):
        return {'mean': self.mean, 'std': self.std}
        
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        return inputs*self.std + self.mean

    def setNormalizationValues(self, mean, std):
        self.mean.assign(mean)
        self.std.assign(std)
        self.set_weights(np.array([mean, std]))
