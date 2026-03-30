import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, nnm, nnc, dataManager):
        self.nnm = nnm
        self.nnc = nnc
        self.dataManager = dataManager

    def plot(self, block=True):


        print('')
        self.nnm.plotLoss(plt, block=False, f=0)
        self.nnc.plotLoss(plt, block=False, f=1)
        self.dataManager.plotStates(plt, block=False, f=2)
        self.dataManager.plotControlInputs(plt, block=False, f=3)
        self.nnm.linearManager.plotEquilibrium(plt, block=False, f=4)
        self.nnm.linearManager.plotEquilibriumFixed(plt, block=False, f=5)
        if not self.nnm.useSparsityLayer:
            self.nnm.linearManager.plotEigVals(plt, block=False, f=6)
        plt.show(block=block)



