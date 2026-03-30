import numpy as np

class Sweep:
    def __init__(self, parSweep, parNnm):
        self.stepSize = parSweep['stepSize']
        self.baseAmplitude = parSweep['amplitude']
        self.amplitude = self.baseAmplitude
        self.amplitudeDecay = parSweep['amplitudeDecay']
        self.nControlInputs = parNnm['nControlInputs']
        self.changeStep()
        self.active = True

    def changeStep(self):
        self.counter = 0
        self.value = np.random.rand(self.nControlInputs)
        self.value = self.value*2 - 1
        self.value = self.value*self.amplitude

    def evaluate(self):
        if not self.active:
            return np.zeros(self.nControlInputs)

        if self.counter >= self.stepSize:
            self.changeStep()

        self.counter += 1
        return self.value
        
    def setActive(self, active):
        self.active = active

    def update(self, i):
        self.amplitude = self.baseAmplitude*(self.amplitudeDecay**i)
