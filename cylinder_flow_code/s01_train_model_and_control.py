from modules.sweep import Sweep
from modules.nnm import Nnm
from modules.nnc import Nnc 
from modules.dataManager import DataManager
from modules.sim import Sim
from modules.plotter import Plotter
import parameters
import os
import numpy as np


print('')

nStepsSweep = parameters.parGeneral['nStepsSweep']
nStepsClosed = parameters.parGeneral['nStepsClosed']
nStepsRelease = parameters.parGeneral['nStepsRelease']
nIterations = parameters.parGeneral['nIterations']

sweep = Sweep(parameters.parSweep, parameters.parNnm)
nnm = Nnm(parameters.parNnm, parameters.parEquilibrium)
nnc = Nnc(nnm, parameters.parNnc)
sim = Sim(sweep, nnc, parameters.parGeneral)
dataManager = DataManager(parameters.parData)
plotter = Plotter(nnm, nnc, dataManager)

if dataManager.isLoadRestart:
    dataManager.loadRestartFiles()
    dataManager.computeNormalization()
    nnc.initializeVariables(dataManager)
    nnm.load(dataManager.restartIndex-1, dataManager.dataDir)
    nnc.load(dataManager.restartIndex-1, dataManager.dataDir)

totalSteps = nIterations*(nStepsSweep+nStepsRelease+nStepsClosed)

if nIterations > 0:
    sim.start(dataManager.t0)

print('')


for it in range(nIterations):
    rstI = dataManager.restartIndex
    i = it + rstI

    print('')
    print('Running iteration {}/{}'.format(i, rstI+nIterations-1))

    sweep.update(i)

    sim.run('sweep')
    dataManager.addResponse(sim)

    if i == 0: 
        nnc.initializeVariables(dataManager)

    nnm.resetWeights()
    nnm.update(dataManager)
    nnc.resetWeights()
    nnc.update(dataManager.gatherData())

    sim.run('release')
    dataManager.addResponse(sim)

    sim.run('closed')
    dataManager.addResponse(sim)

    dataManager.makeSubDir(i)
    dataManager.save(i)
    nnm.save(i, dataManager.dataDir)
    nnc.save(i, dataManager.dataDir)




plotter.plot(block=True)
