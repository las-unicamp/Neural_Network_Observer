import numpy as np
import os
import random
import shutil

class DataManager:
    def __init__(self, parData):

        isLoadRestart = parData['loadRestart']

        self.isLoadRestart = isLoadRestart
        self.dataDir = 'output/' + parData['dataDir']
        self.data = {}
        self.types = ['sweep','release','closed']

        for opt in self.types:
            self.data[opt] = []
            

        if isLoadRestart:
            self.restartIndex = parData['restartIndex']
            self.removeDirs()
        else:   
            self.restartIndex = 0
            self.removeDirs()
            shutil.copy('parameters.py', self.dataDir)
            self.t0 = 0.0


        self.computedNormalization = False

        if not os.path.exists(self.dataDir):
            os.makedirs(self.dataDir)

        self.nnoData = NnoData()

        # For data normalization
        self.normMean = []
        self.normStd = []

        



    def loadRestartFiles(self):

        dirFormat = self.dataDir + '/{:03d}/'
        self.t0 = 0
        for opt in self.types:
            for index in range(self.restartIndex):

                path = dirFormat.format(index)

                file = path + opt + '_time.npy'
                time = np.load(file)

                file = path + opt + '_controlInputs.npy'
                controlInputs = np.load(file)

                file = path + opt + '_states.npy'
                states = np.load(file)

                currentData = DataModule(time, controlInputs, states)
                self.data[opt].append(currentData)

                if time[-1] > self.t0:
                    self.t0 = time[-1]

    def addResponse(self, comm):
        opt = comm.opt
        time = comm.time
        controlInputs = comm.controlInputs
        states = comm.states

        currentData = DataModule(time, controlInputs, states)

        self.data[opt].append(currentData)

        if not self.computedNormalization:
            self.computeNormalization()

    def addNnoData(self, comm):
        controlInputs = comm.controlInputs
        states = comm.states
        outputs = comm.outputs
        self.nnoData.appendData(controlInputs,states,outputs)

        

    def computeNormalization(self):

        data = self.data['sweep'][0]

        self.controlInputsMean = np.mean(data.controlInputs, axis=0)
        self.controlInputsStd = np.std(data.controlInputs, axis=0)
        self.statesMean = np.mean(data.states, axis=0)
        self.statesStd = np.std(data.states, axis=0)
        self.computedNormalization = True
        

    def save(self, index):
        for opt in self.types:
            currentData = self.data[opt][index]
            self.saveSignal(currentData.time, opt, 'time', index)
            self.saveSignal(currentData.controlInputs, opt, 'controlInputs', index)
            self.saveSignal(currentData.states, opt, 'states', index)

    def saveSignal(self, signal, opt, name, index):
        label = opt + "_" + name
        saveStr = self.dataDir + '/{:03d}/' + label + '.npy'
        saveStr = saveStr.format(index)
        np.save(saveStr, signal)
        #print("    Saved: " + saveStr)

    def makeSubDir(self, index):
        dir = self.dataDir + '/{:03d}/'
        dir = dir.format(index)
        os.makedirs(dir)


    def removeDirs(self):


        dir = self.dataDir
        if not os.path.exists(dir):
            os.makedirs(dir)
            
        for file in os.listdir(dir):
            path = os.path.join(dir, file)
            if ('quick_load' in file) or (file == 'debug_signals'):
                continue
            try:
                index = int(file)
                if index >= self.restartIndex:
                    shutil.rmtree(path)
            except:
                try:
                    shutil.rmtree(path)
                except OSError:
                    pass

        
    
    def gatherData(self):

        controlInputs = []
        statesIn = []
        statesOut = []
        
        # Gather all samples and split in output/input for training
        for opt in self.types:
            for currentData in self.data[opt]:
                controlInputs.append(currentData.controlInputs[:-1])
                statesIn.append(currentData.states[:-1])
                statesOut.append(currentData.states[1:])

        # Put samples together in big data arrays
        controlInputs = np.concatenate(controlInputs, axis=0)
        statesIn = np.concatenate(statesIn, axis=0)
        statesOut = np.concatenate(statesOut, axis=0)

        # Put samples in object and shuffle samples
        returnData = TrainDataModule(controlInputs, statesIn, statesOut)

        return returnData

    def plotStates(self, plt, block=True, f=None):
        from matplotlib import pyplot as plt
        color = {"sweep": 'b', "release": 'w', "closed": 'g'}
        plt.figure(f)
        cmap = plt.cm.tab10

        for opt in self.types:
            for i in range(len(self.data[opt])):
                currentData = self.data[opt][i]
                time = currentData.time
                states = currentData.states
                nStates = states.shape[1]
                for j in range(nStates):
                    state = states[:,j]
                    plt.plot(time, state, color=cmap(j % 10))
                c = color[opt]
                dt = time[1] - time[0]
                plt.axvspan(time[0], time[-1]+dt, facecolor=c, alpha=0.1)
        
        plt.show(block=block)

    def plotControlInputs(self, plt, block=True, f=None):
        from matplotlib import pyplot as plt
        color = {"sweep": 'b', "release": 'w', "closed": 'g'}
        plt.figure(f)
        cmap = plt.cm.tab10

        for opt in self.types:
            for i in range(len(self.data[opt])):
                currentData = self.data[opt][i]
                time = currentData.time
                cInps = currentData.controlInputs
                nCInps = cInps.shape[1]
                for j in range(nCInps):
                    cInp = cInps[:,j]
                    plt.plot(time, cInp, color=cmap(j))
                c = color[opt]
                dt = time[1] - time[0]
                plt.axvspan(time[0], time[-1]+dt, facecolor=c, alpha=0.1)
        
        plt.show(block=block)
        
    def getSamples(self, index, opt):
        return self.data[opt][index]


class DataModule:
    def __init__(self, time, controlInputs, states):
        self.time = time
        self.controlInputs = controlInputs
        self.states = states

class TrainDataModule:
    def __init__(self, controlInputs, statesIn, statesOut):
        self.controlInputs = controlInputs
        self.statesIn = statesIn
        self.statesOut = statesOut

    def shuffleData(self):
        indices = [*range(self.controlInputs.shape[0])]
        random.shuffle(indices)
        self.controlInputs = self.controlInputs[indices]
        self.statesIn = self.statesIn[indices]
        self.statesOut = self.statesOut[indices]

    def getRandomSamples(self, nTrain, nTest):

        indices = [*range(self.controlInputs.shape[0])]
        random.shuffle(indices)
        indicesTrain = indices[:nTrain]
        indicesTest = indices[nTrain:nTrain+nTest]


        controlInputsTrain = self.controlInputs[indicesTrain]
        statesInTrain = self.statesIn[indicesTrain]
        statesOutTrain = self.statesOut[indicesTrain]

        controlInputsTest = self.controlInputs[indicesTest]
        statesInTest = self.statesIn[indicesTest]
        statesOutTest = self.statesOut[indicesTest]

        trainData = TrainDataModule(controlInputsTrain,statesInTrain,statesOutTrain)
        testData = TrainDataModule(controlInputsTest,statesInTest,statesOutTest)

        return trainData, testData

class NnoData:
    def __init__(self, controlInputs = None, states = None, outputs = None):
        self.controlInputs = controlInputs
        self.states = states
        self.outputs = outputs

    def appendData(self, controlInputs, states, outputs):
        if self.controlInputs is None:
            self.controlInputs = controlInputs
            self.states = states
            self.outputs = outputs

        else:
            self.controlInputs = np.concatenate([self.controlInputs, controlInputs])
            self.states = np.concatenate([self.states, states])
            self.outputs = np.concatenate([self.outputs, outputs])


    def getRandomSamples(self, nTrain, nTest):

        indices = [*range(self.controlInputs.shape[0])]
        random.shuffle(indices)
        indicesTrain = indices[:nTrain]
        indicesTest = indices[nTrain:nTrain+nTest]


        controlInputsTrain = self.controlInputs[indicesTrain]
        statesTrain = self.states[indicesTrain]
        outputsTrain = self.outputs[indicesTrain]

        controlInputsTest = self.controlInputs[indicesTest]
        statesTest = self.states[indicesTest]
        outputsTest = self.outputs[indicesTest]

        trainData = NnoData(controlInputsTrain,statesTrain,outputsTrain)
        testData = NnoData(controlInputsTest,statesTest,outputsTest)

        return trainData, testData

    def getRandomSamplesOutput(self, nTrain, nTest):

        indices = [*range(self.controlInputs.shape[0]-1)]
        random.shuffle(indices)
        indicesTrain = indices[:nTrain]
        indicesTest = indices[nTrain:nTrain+nTest]

        controlInputs = self.controlInputs[:-1,]
        states = self.states[1:,]
        outputs = self.outputs[1:,]

        controlInputsTrain = controlInputs[indicesTrain]
        statesTrain = states[indicesTrain]
        outputsTrain = outputs[indicesTrain]

        controlInputsTest = controlInputs[indicesTest]
        statesTest = states[indicesTest]
        outputsTest = outputs[indicesTest]

        trainData = NnoData(controlInputsTrain,statesTrain,outputsTrain)
        testData = NnoData(controlInputsTest,statesTest,outputsTest)

        return trainData, testData
    
    def getRandomControlSignals(self, nSamples, nh):
        indices = [*range(self.controlInputs.shape[0]-1)]
        indices = indices[:-nh]
        random.shuffle(indices)

        retSignals = []
        for i in range(nSamples):
            retSignals.append(self.controlInputs[indices[i]:indices[i]+nh])

        return list(np.transpose(np.array(retSignals),axes=(1,0,2)))
    
    def getRandomSignals(self, nSamples, nh):
        indices = [*range(self.controlInputs.shape[0]-1)]
        indices = indices[:-nh]
        random.shuffle(indices)

        retControlSignals = []
        retStateSignals = []
        for i in range(nSamples):
            retControlSignals.append(self.controlInputs[indices[i]:indices[i]+nh])
            retStateSignals.append(self.states[indices[i]:indices[i]+nh])

        controlList = list(np.transpose(np.array(retControlSignals),axes=(1,0,2)))
        statesList = list(np.transpose(np.array(retStateSignals),axes=(1,0,2)))
        return controlList, statesList
            
    def getStatesNormalization(self):
        return np.mean(self.states,axis=0), np.std(self.states,axis=0) 
    
    def getControlInputsNormalization(self):
        return np.mean(self.controlInputs,axis=0), np.std(self.controlInputs,axis=0) 
    
    def getOutputsNormalization(self):
        return np.mean(self.outputs,axis=0), np.std(self.outputs,axis=0) 

