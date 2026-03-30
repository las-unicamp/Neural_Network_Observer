from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras
from modules.nnm import NormalizationLayer
from modules.nnm import DenormalizationLayer
from modules.nnm import SingleConnected
import matplotlib.pyplot as plt

class Nno:
    def __init__(self, nnm, parNno):
        self.nnm =  nnm


        self.nnoDataSize = parNno['nnoDataSize']
        self.trainRatio = parNno['trainRatio']
        self.nMeasurableOutputs = parNno['nMeasurableOutputs']
        self.horizonLength = parNno['horizonLength']
        self.horizonWarmSteps = parNno['horizonWarmSteps']
        self.lossLength = parNno['lossLength']
        self.outputModelLayers = parNno['outputModelLayers']
        self.outputLearnRate = parNno['outputLearnRate']
        self.outputEpochs = parNno['outputEpochs']
        self.outputL2Reg = parNno['outputL2Reg']
        self.observerModelLayers = parNno['observerModelLayers']
        self.observerLearnRate = parNno['observerLearnRate']
        self.observerDecayRate = parNno['observerDecayRate']
        self.observerEpochs = parNno['observerEpochs']
        self.observerL2Reg = parNno['observerL2Reg']
        self.observerCompensationWeight = parNno['observerCompensationWeight']
        self.observerStatesErrorWeight = parNno['observerStatesErrorWeight']
        self.observerWarmupSteps = parNno['observerWarmupSteps']
        self.observerBatchSize = parNno['observerBatchSize']
        self.noiseAmount = parNno['noiseAmount']
        self.noiseAlpha = parNno.get('noiseAlpha', 0.)
        self.stepSkip = parNno['stepSkip']
        self.useDelayedFeedthrough = False # In classic approach, no delayed FT

        if 'activation' in parNno.keys():
            self.activation = parNno['activation']
        else:
            self.activation = 'relu'

        self.estimatedStates = np.ones(self.nnm.nStates)
        self.nnoCompensation = np.zeros(self.nnm.nStates)
        self.curControlInputs = np.zeros(self.nnm.nControlInputs)
        self.skipCounter = 0

        return
    

    def setup(self, nnc):

        self.buildOutputModel()
        self.buildObserverModel()
        self.buildObserverLossFunction(nnc)

    def buildOutputModel(self):

        # NNO parameters
        m = self.nnm.nControlInputs
        n = self.nnm.nStates
        p = self.nMeasurableOutputs

        # Input and normalization layers
        inputs = keras.Input([n])

      
        self.outNormalizationLayer = NormalizationLayer(np.zeros(n), np.ones(n)) 

        y = self.outNormalizationLayer(inputs)

        # Hidden layers
        self.outputDenseLayers = []
        layers = self.outputModelLayers

        for nUnits in layers:
            curDense = keras.layers.Dense(nUnits, activation=self.activation,
                kernel_initializer='random_normal',
                bias_initializer='random_normal')
            
            y = curDense(y)
            self.outputDenseLayers.append(curDense)

        # Output layer
        curDense = keras.layers.Dense(p, activation='linear',
            kernel_initializer='random_normal',
            bias_initializer='random_normal')
        y = curDense(y)
        self.outputDenseLayers.append(curDense)

        # Denormalization layer
        self.outDenormalizationLayer = DenormalizationLayer(np.zeros(p),np.ones(p)) 
        outputs = self.outDenormalizationLayer(y)

        self.outputModel = keras.Model(inputs, outputs)

        # Build loss function
        modelOut = self.outputModel.output
        self.outLabels = tf.placeholder(tf.float32, shape=modelOut.shape)

        self.tfOutStds = tf.Variable(np.zeros(p),dtype=tf.float32)
        self.tfStatesStds = tf.Variable(np.zeros(n),dtype=tf.float32)


        lossErr = tf.square((self.outLabels-modelOut)/self.tfOutStds)
        lossErr = tf.reduce_mean(lossErr, axis=-1)
        lossErr = tf.reduce_mean(lossErr, axis=-1)

        lossRegL2 = 0
        for curDense in self.outputDenseLayers:
            curWeights = curDense.weights[0]
            lossRegL2 = lossRegL2 + tf.reduce_sum(tf.square(curWeights))

        lossTotal = lossErr + self.outputL2Reg*lossRegL2

        self.outputLoss = lossTotal
        self.outLossErr = lossErr
        self.outLossRegL2 = lossRegL2


        # Get trainable weights
        trainableVars = []
        for layer in self.outputDenseLayers:
            weights = layer.weights
            for w in weights:
                trainableVars.append(w)

        # Optimizer
        lRate = self.outputLearnRate
        optimizer = tf.train.AdamOptimizer(learning_rate=lRate)
        optimizer = optimizer.minimize(lossTotal,var_list=trainableVars)
        self.outOptimizer = optimizer

    def buildObserverModel(self):

        # Parameters
        m = self.nnm.nControlInputs
        n = self.nnm.nStates
        p = self.nMeasurableOutputs


        # Input and normalization layers
        inputs1 = keras.Input([p])
        inputs2 = keras.Input([p])
        self.obsNormalizationLayer = NormalizationLayer(np.zeros(p), np.ones(p)) 
        v1 = self.obsNormalizationLayer(inputs1)
        v2 = self.obsNormalizationLayer(inputs2)
        v = keras.layers.Concatenate()([v1,v2])

        # Hidden layers
        self.observerDenseLayers = []
        layers = self.observerModelLayers

        for nUnits in layers:
            curDense = keras.layers.Dense(nUnits, activation=self.activation,
                kernel_initializer='random_normal',
                bias_initializer='random_normal')
            
            v = curDense(v)
            self.observerDenseLayers.append(curDense)

        # Output layer
        curDense = keras.layers.Dense(n, activation='linear',
            kernel_initializer='random_normal',
            bias_initializer='random_normal')
        v = curDense(v)
        self.observerDenseLayers.append(curDense)

        # Denormalization layer
        self.obsDenormalizationLayer = DenormalizationLayer(np.zeros(n), np.ones(n))
        outputs = self.obsDenormalizationLayer(v)

        self.observerModel = keras.Model([inputs1, inputs2], outputs)

    def buildObserverLossFunction(self, nnc):


        # Parameters
        m = self.nnm.nControlInputs
        n = self.nnm.nStates
        p = self.nMeasurableOutputs
        nh = self.horizonLength
        l2reg = self.observerL2Reg
        wv = self.observerCompensationWeight
        wx = self.observerStatesErrorWeight
        stepSkip = self.stepSkip

        # Used models
        outputModel = self.outputModel
        nnm = self.nnm.model
        nnc = nnc.model
        observerModel = self.observerModel

        # Training model inputs
        inpsX = keras.Input([n])
        inpsXGuess = keras.Input([n])
        inpsV = keras.Input([n])

        # Open loop control
        inpsU = [keras.Input([m]) for _ in range(nh)]

        # Measurement noise
        inpsNoise = [keras.Input([p]) for _ in range(nh)]
        
        # Current states
        curX = inpsX
        curXGuess = inpsXGuess
        curV = inpsV

        # List of output errors and states compensation for building loss
        errorList = []
        statesErrorList = []
        vList = []

        # For skipping measurements
        skipCounter = 0

        # TARSUS For debugging
        self.debY = []
        self.debYEst = []
        self.debV = []

        # Loop through finite horizon
        for i in range(nh):

            # Evaluate control inputs
            curU = nnc(curX) + inpsU[i]

            # Compute outputs a priori (classic approach)
            curY = outputModel(curX)
            curYGuess = outputModel(curXGuess)

            # Update actual states
            curUX = keras.layers.Concatenate()([curU, curX])
            curX = nnm(curUX)

            # Update guessed states
            curUXGuess = keras.layers.Concatenate()([curU, curXGuess])
            curXGuess = nnm(curUXGuess)

            curYNoisy = curY+inpsNoise[i]
            
            errorList.append(curY-curYGuess)
            statesErrorList.append(curX-curXGuess)

            #TARSUS For debugging
            self.debY.append(curY)
            self.debYEst.append(curYGuess)

            # Compute states compensation
            if skipCounter == stepSkip:
                skipCounter = 0
            if skipCounter == 0:
                curV = observerModel([curYNoisy,curYGuess])
            
            # Estimated states correction
            curXGuess = keras.layers.Add()([curXGuess,curV])

            skipCounter += 1
                                
            self.debV.append(curV)
            vList.append(curV)

        # The last V is irrelevant
        vList.pop()

        # Keep track of the input tensors
        self.trainInpStates = inpsX
        self.trainInpStatesGuess = inpsXGuess
        self.trainInpCompensations = inpsV
        self.trainOpenSignal = inpsU
        self.trainNoise = inpsNoise

        # Stacking variables for loss function
        outputErrors = tf.stack(errorList,axis=1)
        statesErrors = tf.stack(statesErrorList,axis=1)
        stateCompensations = tf.stack(vList,axis=1)

        # Build loss
        self.obsOutputErrors = outputErrors
        errorLoss = []
        statesErrorLoss = []
        compensationLoss = []
        optimizer = []
        self.truncatedLoss = []


        regL2Loss = 0
        for curDense in self.observerDenseLayers:
            curWeights = curDense.weights[0]
            regL2Loss = regL2Loss + tf.reduce_sum(tf.square(curWeights))

        self.obsRegL2Loss = regL2Loss


        # Get trainable weights
        trainableVars = []
        for layer in self.observerDenseLayers:
            weights = layer.weights
            for w in weights:
                trainableVars.append(w)

        # Setup for training the observer weightss
        self.tfObsLearnRate = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)
        adamOptimizer = tf.train.AdamOptimizer(learning_rate=self.tfObsLearnRate)

        nWarm = len(self.horizonWarmSteps)
        for i in range(nWarm+1):
            start = nh-nWarm+i-self.lossLength
            start = max(0,start)
            curErrorLoss = tf.square(outputErrors[:,start:nh-nWarm+i,:]/self.tfOutStds)
            curErrorLoss = tf.reduce_mean(curErrorLoss, axis=-1)
            curErrorLoss = tf.reduce_mean(curErrorLoss, axis=-1)
            curErrorLoss = tf.reduce_mean(curErrorLoss, axis=-1)
            errorLoss.append(curErrorLoss)

            curStatesErrorLoss = tf.square(statesErrors[:,start:nh-nWarm+i,:]/self.tfStatesStds)
            curStatesErrorLoss = tf.reduce_mean(curStatesErrorLoss, axis=-1)
            curStatesErrorLoss = tf.reduce_mean(curStatesErrorLoss, axis=-1)
            curStatesErrorLoss = tf.reduce_mean(curStatesErrorLoss, axis=-1)
            statesErrorLoss.append(curStatesErrorLoss)

            curCompensationLoss = tf.square(stateCompensations[:,start:nh-nWarm+i,:]/self.tfStatesStds)
            curCompensationLoss = tf.reduce_mean(curCompensationLoss, axis=-1)
            curCompensationLoss = tf.reduce_mean(curCompensationLoss, axis=-1)
            curCompensationLoss = tf.reduce_mean(curCompensationLoss, axis=-1)
            compensationLoss.append(curCompensationLoss)

            curLoss = curErrorLoss + wx*curStatesErrorLoss + wv*curCompensationLoss + l2reg*regL2Loss
            self.truncatedLoss.append(curLoss)
            optimizer.append(adamOptimizer.minimize(curLoss, var_list=trainableVars))

        self.observerLoss = curErrorLoss + wx*curStatesErrorLoss + wv*curCompensationLoss + l2reg*regL2Loss
        self.obsErrLoss = curErrorLoss
        self.obsStatesLoss = curStatesErrorLoss
        self.obsCompLoss = curCompensationLoss


        self.obsOptimizer = optimizer

        return

    def trainOutputModel(self, nnoData):
        
        # Training parameters
        nh = self.horizonLength
        nTrain = round(self.nnoDataSize*self.trainRatio)
        nTest = self.nnoDataSize - nTrain

        # Initialize normalization layers
        statesMean, statesStd = nnoData.getStatesNormalization()
        controlInputsMean, controlInputsStd = nnoData.getControlInputsNormalization()
        outputsMean, outputsStd = nnoData.getOutputsNormalization()

        meanIn = statesMean[:]
        stdIn = statesStd[:]

        self.outNormalizationLayer.setNormalizationValues(meanIn, stdIn)

        meanOut = np.array(outputsMean)
        stdOut = np.array(outputsStd)
        self.outDenormalizationLayer.setNormalizationValues(meanOut, stdOut)

        
        self.tfOutStds.assign(outputsStd)

        # Get data for training
        trainData, testData = nnoData.getRandomSamplesOutput(nTrain, nTest)


        trainDataIn = trainData.states[:]
        testDataIn = testData.states[:]
        trainDataOut = np.array(trainData.outputs)
        testDataOut = np.array(testData.outputs)


        # Train the ouput model
        optimizer = self.outOptimizer
        printStr = "    Training output model: {}/{}, errLoss: {:.4e}, reg2Loss: {:.4e}, "+\
            "totalLoss: {:.4e}, testLoss: {:.4e}     "
        
        bestWeights = self.outputModel.get_weights()
        bestLoss = float('inf')
        sess = self.nnm.sess

        with sess.as_default():
            for i in range(self.outputEpochs):

                if i==0:
                    sess.run(self.tfStatesStds.assign(statesStd))
                    sess.run(self.tfOutStds.assign(outputsStd))
                    
                optimizer.run(feed_dict={
                self.outputModel.input: trainDataIn, 
                self.outLabels: trainDataOut})

                lossErrVal = sess.run(self.outLossErr,feed_dict={
                self.outputModel.input: trainDataIn,
                self.outLabels: trainDataOut})

                lossRegL2Val = sess.run(self.outLossRegL2,feed_dict={
                self.outputModel.input: trainDataIn,
                self.outLabels: trainDataOut})

                lossTotalVal = sess.run(self.outputLoss,feed_dict={
                self.outputModel.input: trainDataIn,
                self.outLabels: trainDataOut})

                lossTestVal = sess.run(self.outputLoss,feed_dict={
                self.outputModel.input: testDataIn,
                self.outLabels: testDataOut})

                print(printStr.format(
                    i+1, self.outputEpochs, 
                    lossErrVal, lossRegL2Val, lossTotalVal, lossTestVal), 
                    end='\r')

                if bestLoss > lossTotalVal:
                    bestWeights = self.outputModel.get_weights()
                    bestLoss = lossTotalVal

        self.outputModel.set_weights(bestWeights)

        print(printStr.format(
            i+1, self.outputEpochs, 
            lossErrVal, lossRegL2Val, lossTotalVal, lossTestVal))
        
        # Keep track of normalization values for the observer model
        self.statesMean = statesMean
        self.statesStd = statesStd
        self.controlInputsMean = controlInputsMean
        self.controlInputsStd = controlInputsStd
        self.outputsMean = outputsMean
        self.outputsStd = outputsStd

        # Set initial condition for observer
        self.estimatedStates = self.statesMean

    def trainObserverModel(self, nnoData):

        # Normalization parameters
        statesMean = self.statesMean
        statesStd = self.statesStd
        outputsMean = self.outputsMean
        outputsStd = self.outputsStd

        # Update normalization layers
        meanIn = np.array(outputsMean)
        stdIn = np.array(outputsStd)
        self.obsNormalizationLayer.setNormalizationValues(meanIn, stdIn)

        meanOut = np.array(statesMean) # TARSUS maybe this is not the most apropriate normalization vals
        stdOut = np.array(statesStd)
        self.obsDenormalizationLayer.setNormalizationValues(meanOut, stdOut)

        # Parameters
        nTrain = round(self.nnoDataSize*self.trainRatio)
        nTest = self.nnoDataSize - nTrain
        m = self.nnm.nControlInputs
        n = self.nnm.nStates
        p = self.nMeasurableOutputs
        nh = self.horizonLength


        optimizer = self.obsOptimizer

        # Get data for training
        trainData, testData = nnoData.getRandomSamples(nTrain, nTest)

        trainDataStates = trainData.states
        trainDataStatesGuess = trainDataStates[np.random.permutation(trainDataStates.shape[0])]
        trainDataCompensations = np.zeros(trainDataStates.shape) # TARSUS I'm trying with zeros; maybe random?
        testDataStates = testData.states
        testDataStatesGuess = testDataStates[np.random.permutation(testDataStates.shape[0])]
        testDataCompensations = np.zeros(testDataStates.shape) # TARSUS I'm trying with zeros; maybe random?

        # Get Open loop control signals
        trainOpenControl = nnoData.getRandomControlSignals(nTrain,nh)
        testOpenControl = nnoData.getRandomControlSignals(nTest,nh)

        # Train the observer model
        printStr = "    Training observer model: {}/{}, errLoss: {:.4e}, statesLoss {:.4e}, compLoss: {:.4e}, "+\
            "regL2Loss: {:.4e}, totalLoss: {:.4e}, testLoss: {:.4e}     "

        bestWeights = self.observerModel.get_weights()
        bestLoss = float('inf')
        sess = self.nnm.sess

        # Feed dictionaries
        trainFd = {
                self.trainInpStates: trainDataStates,
                self.trainInpStatesGuess: trainDataStatesGuess, 
                self.trainInpCompensations: trainDataCompensations}
        trainFd.update({self.trainOpenSignal[k]: trainOpenControl[k] for k in range(len(trainOpenControl))})
        trainNoiseSeq = self.genColorNoise(nTrain,p,self.noiseAmount,self.noiseAlpha,nh)
        trainFd.update({self.trainNoise[k]: trainNoiseSeq[k] for k in range(len(self.trainNoise))})

        testFd = {
                self.trainInpStates: testDataStates,
                self.trainInpStatesGuess: testDataStatesGuess, 
                self.trainInpCompensations: testDataCompensations}
        testFd.update({self.trainOpenSignal[k]: testOpenControl[k] for k in range(len(testOpenControl))})
        testNoiseSeq = self.genColorNoise(nTest,p,self.noiseAmount,self.noiseAlpha,nh)
        testFd.update({self.trainNoise[k]: testNoiseSeq[k] for k in range(len(self.trainNoise))})
        
        # Initialize lists to store loss values
        totalLosses = []
        testLosses = []
        truncatedLosses = []

        # Set up the plot
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        line1, = ax.plot([], [], label='Total Loss', color='blue')
        line2, = ax.plot([], [], label='Test Loss', color='red')
        line3, = ax.plot([], [], label='Truncated Loss', color='green')
        ax.legend()
        ax.set_yscale('log')

        forceStop = False
        optCounter = 0
        hws = self.horizonWarmSteps
        nWarm = len(hws)

        with sess.as_default():
            for i in range(self.observerEpochs):
                if optCounter<nWarm:
                    if i>hws[optCounter]:
                        optCounter = optCounter+1
                opt=optimizer[optCounter]
                warmupSteps = self.observerWarmupSteps
                if i<warmupSteps:
                    tf.keras.backend.set_value(self.tfObsLearnRate, self.observerLearnRate*i/warmupSteps)
                else:
                    tf.keras.backend.set_value(self.tfObsLearnRate, self.observerLearnRate*self.observerDecayRate**(i-warmupSteps))

                for j in range(int(nTrain/self.observerBatchSize)+1):
                    opt.run(feed_dict=self.getBatch(trainFd,j))
                errLoss = sess.run(self.obsErrLoss, feed_dict=trainFd)
                statesLoss = sess.run(self.obsStatesLoss, feed_dict=trainFd)
                compLoss =  sess.run(self.obsCompLoss, feed_dict=trainFd)
                regL2Loss =  sess.run(self.obsRegL2Loss, feed_dict=trainFd)
                totalLoss =  sess.run(self.observerLoss, feed_dict=trainFd)
                testLoss =  sess.run(self.observerLoss, feed_dict=testFd)
                truncatedLoss = sess.run(self.truncatedLoss[optCounter], feed_dict=testFd)

                totalLosses.append(totalLoss)
                testLosses.append(testLoss)
                truncatedLosses.append(truncatedLoss)

                print(printStr.format(
                    i+1, self.observerEpochs, 
                    errLoss,statesLoss,compLoss,regL2Loss,totalLoss,testLoss), 
                    end='\r')
                

                # Update the plot every 20 iterations
                if (i + 1) % 20 == 0:
                    # Update the plot
                    line1.set_data(range(len(totalLosses)), totalLosses)
                    line2.set_data(range(len(testLosses)), testLosses)
                    line3.set_data(range(len(truncatedLosses)), truncatedLosses)
                    
                    x_data, y_data = line3.get_data()
                    ax.set_xlim(min(x_data), max(x_data))  # Update x-axis limits based on line3
                    ax.set_ylim(min(y_data), max(y_data))

                    plt.draw()
                    plt.pause(0.01)
                if bestLoss > totalLoss:
                    bestWeights = self.observerModel.get_weights()
                    bestLoss = totalLoss
                
                if forceStop:
                    break

        self.observerModel.set_weights(bestWeights)
        
        # TARSUS for debugging
        debY =  [sess.run(curY, feed_dict=testFd) for curY in self.debY]
        
        debYEst =  [sess.run(curYEst, feed_dict=testFd) for curYEst in self.debYEst]
        
       
        debV =  [sess.run(curV, feed_dict=testFd) for curV in self.debV]
        
        debYErr =  sess.run(self.obsOutputErrors, feed_dict=testFd)
        

        print(printStr.format(
            i+1, self.observerEpochs, 
            errLoss,statesLoss,compLoss,regL2Loss,totalLoss,testLoss), 
            end='\r')
        print('')

    def eval(self, outputs):
        # Signals
        y = outputs
        xEst = self.estimatedStates
        u = self.curControlInputs

        # Predict output
        yEst = self.outputModel.predict([[xEst]])[0]

        # Compute compensation
        if self.skipCounter == self.stepSkip:
            self.skipCounter = 0
        if self.skipCounter == 0:
            v = self.observerModel.predict([[y],[yEst]])[0] 
        else:
            v = self.nnoCompensation
        self.skipCounter += 1

        # Predict states
        uxEst = np.concatenate([u,xEst])
        xEst = self.nnm.model.predict([[uxEst]])[0]
        xEst = xEst+v

        self.estimatedStates = xEst
        self.nnoCompensation = v
        self.obsOutputs = yEst

    def getBatch(self,dataDic,position):
        batchSize = self.observerBatchSize
        dataSize = dataDic[self.trainInpStates].shape[0]

        start = position*batchSize
        end = min((position+1)*batchSize, dataSize)

        batch = {
                self.trainInpStates: dataDic[self.trainInpStates][start:end],
                self.trainInpStatesGuess: dataDic[self.trainInpStatesGuess][start:end],
                self.trainInpCompensations: dataDic[self.trainInpCompensations][start:end],
                }
        batch.update({self.trainOpenSignal[k]: dataDic[self.trainOpenSignal[k]][start:end] for k in range(len(self.trainOpenSignal))})
        batch.update({self.trainNoise[k]: dataDic[self.trainNoise[k]][start:end] for k in range(len(self.trainNoise))})
        
        return batch
    
    def genColorNoise(self,nTrain,p,noiseAmount,noiseAlpha,nh):
        noise = np.zeros((nh, nTrain, p))
        
        # Initial state: randomly sampled from stationary distribution
        noise[0] = np.random.normal(loc=0, scale=noiseAmount, size=(nTrain, p))

        for k in range(1, nh):
            whiteNoise = np.random.normal(loc=0, scale=noiseAmount, size=(nTrain, p))
            noise[k] = noiseAlpha * noise[k - 1] + np.sqrt(1 - noiseAlpha**2) * whiteNoise

        # Convert into list of nh arrays (each of shape (nTrain, p))
        return [noise[k] for k in range(nh)]


