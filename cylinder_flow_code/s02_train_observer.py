from modules.sweep import Sweep
from modules.nnm import Nnm
from modules.nnc import Nnc 
from modules.nno_classic import Nno 
from modules.dataManager import DataManager
from modules.sim import Sim
from modules.plotter import Plotter
from modules.filters import ButterworthHighPassFilter
import numpy as np
import parameters
import os


print('')

nStepsSweep = parameters.parGeneral['nStepsSweep']
nStepsClosed = parameters.parGeneral['nStepsClosed']
nStepsRelease = parameters.parGeneral['nStepsRelease']
nIterations = parameters.parGeneral['nIterations']

sweep = Sweep(parameters.parSweep, parameters.parNnm)
nnm = Nnm(parameters.parNnm, parameters.parEquilibrium)
nnc = Nnc(nnm, parameters.parNnc)
nno = Nno(nnm, parameters.parNno)
sim = Sim(sweep, nnc, parameters.parGeneral, nno=nno)
dataManager = DataManager(parameters.parData)
plotter = Plotter(nnm, nnc, dataManager)

if dataManager.isLoadRestart:
    dataManager.loadRestartFiles()
    dataManager.computeNormalization()
    nno.setup(nnc)
    nnc.initializeVariables(dataManager)
    nnm.load(dataManager.restartIndex-1, dataManager.dataDir)
    nnc.load(dataManager.restartIndex-1, dataManager.dataDir)

totalSteps = 3*nStepsSweep + 2*nStepsClosed

if nIterations > 0:
    sim.start(dataManager.t0)

alpha = 1
wandb = nnc.model.layers[-2].get_weights()
wandb[0] = wandb[0]/alpha
wandb[1] = wandb[1]/alpha
nnc.model.layers[-2].set_weights(wandb)
sim.nncGain = alpha

print('')


it = 0
rstI = dataManager.restartIndex
i = it + rstI

quickLoad = False

if quickLoad:
    controlInputsData = np.loadtxt('output/'+parameters.parData['dataDir']+'/quick_load/q_controlInputs.txt')
    if len(controlInputsData.shape)==1:
        controlInputsData = controlInputsData.reshape((controlInputsData.shape[0],1))
    statesData = np.loadtxt('output/'+parameters.parData['dataDir']+'/quick_load/q_states.txt')
    outputsData = np.loadtxt('output/'+parameters.parData['dataDir']+'/quick_load/q_outputs.txt')
    dataManager.nnoData.appendData(controlInputsData,statesData,outputsData)

else:
    print('')
    print('Running iteration {}/{}'.format(i, rstI+nIterations-1)) 

    nnc.ignore = False
    sim.run('sweep')
    dataManager.addNnoData(sim)

    nnc.ignore = True
    sim.run('sweep')
    dataManager.addNnoData(sim)

    nnc.ignore = False
    sim.run('closed')
    dataManager.addNnoData(sim)


# Train and finish
nno.trainOutputModel(dataManager.nnoData)
nno.trainObserverModel(dataManager.nnoData)

nnc.ignore = True
sim.run('sweep',useNno=True)
dataManager.addNnoData(sim)

xEst = sim.estimatedStates
v = sim.compSignals
x = sim.states
outs_real = sim.outputs
outs_obsv = sim.obsOutputs

nnc.ignore = False
sim.run('closed',useNno=True)
dataManager.addNnoData(sim)
