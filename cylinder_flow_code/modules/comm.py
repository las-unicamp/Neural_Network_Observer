import numpy as np
import os
from time import sleep


class Comm:

    def __init__(self, sweep, nnc, parGeneral, nno=None):

        self.sweep = sweep
        self.nnc = nnc
        self.nStepsSweep = parGeneral['nStepsSweep']
        self.nStepsRelease = parGeneral['nStepsRelease']
        self.nStepsClosed = parGeneral['nStepsClosed']
        self.controlSaturation = np.abs(parGeneral['controlSaturation'])
        self.writePath = 'comm/nn_data.txt'
        self.readPath = 'comm/flow_data.txt'
        self.filters = None
        if 'estimatorFactor' in parGeneral.keys():
            self.estimatorFactor = parGeneral['estimatorFactor']
        else:
            self.estimatorFactor = 0
        self.statesMem = np.zeros(self.nnc.nnm.nStates)
        self.controlMem = np.zeros(self.nnc.nnm.nControlInputs)
        self.filteredStatesList = []
        self.measuredStatesList = []
        self.nncGain = 1
        self.nno = nno

        self.delayedOutputs = None

        self.debX = []
        self.debXEst = []

        if os.path.exists(self.writePath):
            os.remove(self.writePath)


    def run(self, opt, useNno=False):

        self.time = []
        self.controlInputs = []
        self.states = []
        self.outputs = []
        self.obsOutputs = []
        self.estimatedStates = []
        self.compSignals = []
        self.opt = opt

        
        n = self.nnc.nnm.nStates
        nit = self.setupComm()

        self.sweep.changeStep()

        print('  Running mode: ' + opt)
        for i in range(nit):

            print('  {}/{}'.format(i+1, nit), end='\r')

            while os.path.exists(self.writePath):
                sleep(0.0015)

            readData = np.loadtxt(self.readPath)

            self.time.append(readData[0:1,])
            readStates = readData[1:n+1,]
            self.states.append(readStates)
            curOutputs = readData[n+1:,]
            self.outputs.append(curOutputs)

            if self.delayedOutputs is None:
                self.delayedOutputs = curOutputs

            if useNno:
                self.nno.eval(self.delayedOutputs)
                self.obsOutputs.append(self.nno.obsOutputs)
                self.compSignals.append(self.nno.nnoCompensation)

            if (not useNno):
                states = readStates
            else:
                states = self.nno.estimatedStates
                self.estimatedStates.append(states)
                self.debX.append(readStates)
                self.debXEst.append(states)


            self.delayedOutputs = curOutputs
            curControlInputs = self.evalControl(states) 

            self.controlInputs.append(curControlInputs)
            
            if useNno:
                self.nno.curControlInputs = curControlInputs


            np.savetxt(self.writePath, self.controlInputs[-1])
            os.remove(self.readPath)


        self.time = np.array(self.time)
        self.controlInputs = np.array(self.controlInputs)
        self.states = np.array(self.states)
        self.outputs = np.array(self.outputs)



        print('')
        
    def setupComm(self):

        if self.opt == 'release':
            self.sweep.setActive(False)
            self.nnc.setActive(False)
            nit = self.nStepsRelease
            
        elif self.opt == 'sweep':
            self.sweep.setActive(True)
            self.nnc.setActive(True)
            nit = self.nStepsSweep

        elif self.opt == 'closed':
            self.sweep.setActive(False)
            self.nnc.setActive(True)
            nit = self.nStepsClosed

        return nit

    def applyEstimator(self, measuredStates):
        alpha = self.estimatorFactor
        
        ux = np.concatenate((self.controlMem,self.statesMem))
        predictedStates = self.nnc.nnm.model.predict(np.array([ux]))[0]

        statesError = measuredStates - predictedStates
        comp = statesError*alpha

        filteredStates = predictedStates + comp


        return filteredStates


    def evalControl(self, measuredStates):
        
        # Open-loop actuation
        uSweep = self.sweep.evaluate()

        if self.estimatorFactor > 0 and self.nnc.trained: 
            filteredStates = self.applyEstimator(measuredStates)
        else:
            filteredStates = measuredStates
            
        # Closed-loop actuation
        uNnc = self.nnc.evaluate(filteredStates)*self.nncGain
        if self.filters is not None:
            for i in range(len(self.filters)):
                uNnc[i] = self.filters[i].apply(uNnc[i])

        # Saturation
        uTotal = uSweep + uNnc
        uSaturated = uTotal
        satVal = self.controlSaturation
        uSaturated = [np.min((val, +satVal*self.nncGain)) for val in uSaturated]
        uSaturated = [np.max((val, -satVal*self.nncGain)) for val in uSaturated]
        uSaturated = np.array(uSaturated)

        self.controlMem = uSaturated
        self.statesMem = filteredStates


        self.filteredStatesList.append(filteredStates)
        self.measuredStatesList.append(measuredStates)

        return uSaturated
    

    def close(self):
        np.savetxt(self.writePath, [])

    def start(self, t0, nSteps):
        if isinstance(object, list):
            t0 = t0[0]
        np.savetxt(self.writePath, [t0, nSteps])
        os.remove(self.readPath)

        while os.path.exists(self.writePath):
            sleep(0.0015)

    


