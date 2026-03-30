# General parameters
parGeneral = {
    'nIterations': 15,
    'nStepsSweep': 2000,
    'nStepsClosed': 300,
    'nStepsRelease': 400,
    'controlSaturation': 0.002
}

# Open-loop sweeep parameters
parSweep = {
    'stepSize': 10,
    'amplitude': parGeneral['controlSaturation'],
    'amplitudeDecay': 0.8
}

# NNM and training parameters
parNnm = {
    'nnmDataSize': 4000,
    'learnRate': 0.006,
    'learnEpochs': 1000,
    'nnmLayers': [18],
    'trainRatio': 0.7,
    'hiddenLayersL2Reg': 1e-5,
    'useSparsityLayer': True,
    'sparsityLayerL1Reg': 1e-4,
    'sparsityTolerance': 3e-3,
    'nControlInputs': 3,
    'nStates': 60
}

# NNC and training parameters
parNnc = {
    'nncDataSize': 1000,
    'learnRate': 0.01,
    'learnEpochs': 200,
    'horizonLength': 50,
    'trainRatio': 0.90,
    'nncLayers': [8],
    'controlSaturation': parGeneral['controlSaturation'],
    'controlInputsWeight': 0.005
}

# NNO and training parameters
parNno = {
    'nnoDataSize': 2800,
    'trainRatio': 0.90,
    'nMeasurableOutputs': 3,
    'horizonLength': 18,
    'horizonWarmSteps': list(range(600,4500,300)),
    'horizonWarmSkip': 1,
    'lossLength': 100,
    'outputModelLayers': parNnm['nnmLayers'],
    'outputLearnRate': parNnm['learnRate'],
    'outputEpochs': 6000,
    'outputL2Reg': 1e-7,
    'observerModelLayers': [18,18],
    'observerLearnRate': 0.015,
    'observerDecayRate': 1.0 + 0*.999,
    'observerEpochs': 6000,
    'observerL2Reg': 1e-7,
    'observerCompensationWeight': 2,
    'observerStatesErrorWeight': 0.05,
    'observerWarmupSteps': 20,
    'observerBatchSize': 3000,
    'noiseAmount': 0.00,
    'noiseAlpha': 0.0,
    'useDelayedFeedthrough': False,
    'stepSkip': 1
}

# Equilibrium and linear analysis
parEquilibrium = {
    'initialGuess': [.1 for _ in range(60)],
    'nNewtonSteps': 10
}

# Data loading parameters
parData = {
    'loadRestart': False,
    'dataDir': 'kuramoto',
    'restartIndex': 10
}
