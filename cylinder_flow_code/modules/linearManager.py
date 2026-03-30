import numpy as np
import tensorflow as tf
from tensorflow import keras

class LinearManager:
    def __init__(self, initialEquilibrium, nNewtonSteps):

        self.equilibrium = np.array(initialEquilibrium)
        self.equilibriumLog = []
        self.eqFixLog = []
        self.sparsityLog = []
        self.eigValLog = []
        self.gradListA = []
        self.gradListB = []
        self.gradDone = False

        self.nNewtonSteps = nNewtonSteps
        self.logEquilibrium([self.equilibrium])

    def logEquilibrium(self, eqList):
        self.equilibriumLog.append(eqList)

    def logEqFix(self, eqFix):
        self.eqFixLog.append(eqFix)

    def logSparsity(self, ids):
        self.sparsityLog.append(ids)

    def logEigVal(self, eigVal):
        self.eigValLog.append(eigVal)

    def update(self, nnm, dataManager):
        self.linearizeModelBp(nnm)
        self.linearizeModelReg(nnm, dataManager)
        self.updateEquilibrium(nnm)
        self.updateEigen(nnm)

    def linearizeModelReg(self, nnm, dataManager):
        
        data = dataManager.getSamples(-1, 'sweep')
        _, idsF = nnm.sparsityLayer.getSparseMap(opt='full')
        _, idsS = nnm.sparsityLayer.getSparseMap(opt='states')
        m = nnm.nControlInputs
        n = nnm.nStates

        states = data.states
        controlInputs = data.controlInputs
        inps = np.concatenate((controlInputs[:-1],states[:-1]), axis=1)
        inps = list(inps)
        statesOut = states[1:]

        for i in range(len(inps)):
            inps[i] = inps[i][idsF]

        inps = np.array(inps)

        pinv = np.linalg.pinv

        def smConv(eq):
            
            eqF = np.concatenate((eq[idsF], [0]))
            eq = np.concatenate((eq, [0]))
            nMeas = inps.shape[0]
            inpsAug = np.concatenate((inps, np.ones((nMeas,1))), axis=1)
            mat = np.matmul((statesOut-eq[m:-1]).T, pinv(inpsAug-eqF).T)[:,m:]
            a = mat[:,:-1]
            return a

        def smRed(eq):

            eqF = np.concatenate((eq[idsF], [0]))
            eq = np.concatenate((eq, [0]))
            nMeas = inps.shape[0]
            inpsAug = np.concatenate((inps, np.ones((nMeas,1))), axis=1)
            mat = np.matmul((statesOut-eq[m:-1]).T, pinv(inpsAug-eqF).T)[idsS,m:]
            a = mat[:,:-1]
            c = mat[:,-1]
            ns = len(idsS)
            eqFix = np.linalg.inv(np.identity(ns) - a)
            eqFix = np.matmul(eqFix,c)
            return a, eqFix


        self.stateMatrixConvReg = smConv
        self.stateMatrixReducedReg = smRed

    def linearizeModelBp(self, nnm):
        
        # NNM trained model
        model = nnm.model
        m = nnm.nControlInputs
        n = nnm.nStates

        # Get the relevant positions acording to nonzero values in sparsity layer
        mask, ids = nnm.sparsityLayer.getSparseMap(opt='states')
        self.logSparsity(ids)

        jacA = []
        jacB = []
        jacFullA = []
        jacFullB = []
        printStr = "    Linearizing NNM model ({}/{})"

        # Compute Jacobian
        for i in range(n):
            print(printStr.format(i+1,n), end='\r')

            # Compute gradients
            if not self.gradDone:
                gradFunc = tf.gradients(model.output[:,i], model.input)
                gradFunc = tf.reshape(gradFunc[0],[-1,1,m+n])

                # The 'm' leftmost columns of this tensor corresponds to matrix 'B'. The rest is 'A'.
                gradFuncA = gradFunc[:,:,m:]
                self.gradListA.append(gradFuncA)
                gradFuncB = gradFunc[:,:,:m]
                self.gradListB.append(gradFuncB)

            gradFuncA = self.gradListA[i]
            gradFuncA = tf.gather(gradFuncA,ids,axis=2)
            gradFuncB = self.gradListB[i]

            # Append to Jacobians for two different linearizations, explained below
            jacFullA.append(gradFuncA)
            jacFullB.append(gradFuncB)
            if mask[i] == 1:
                jacA.append(gradFuncA)
                jacB.append(gradFuncB)

        # ns x ns matrix, where ns is the number of relevant variables after sparsity layer
        self.stateMatrixReduced = jacA
        self.stateMatrixReduced = tf.concat(self.stateMatrixReduced,1)

        # ns x m matrix, where ns is the number of relevant variables after sparsity layer
        self.controlMatrixReduced = jacB
        self.controlMatrixReduced = tf.concat(self.controlMatrixReduced,1)


        # n x ns matrix, which can be used to convert a computed 
        # eigenvector matrix back to the original number of states
        self.stateMatrixConv = jacFullA
        self.stateMatrixConv = tf.concat(self.stateMatrixConv,1)

        # n x m matrix
        self.controlMatrixConv = jacFullB
        self.controlMatrixConv = tf.concat(self.controlMatrixConv,1)

        print('')

        self.gradDone = True

        return

    def updateEigen(self, nnm):
        
        # NNM trained model
        model = nnm.model
        m = nnm.nControlInputs
        n = nnm.nStates
        f = model.predict

        # Number of relevant variables from sparsity procedure
        mask, ids = nnm.sparsityLayer.getSparseMap(opt='states')
        nr = int(np.sum(mask))

        # State matrices
        aRed = self.stateMatrixReducedVal
        aConv = self.stateMatrixConvVal

        # Eigenvalues and eigenvectors for reduced system
        eVal, eVec = np.linalg.eig(aRed)
        idEs = np.abs(eVal).argsort()[::-1]
        eVal = eVal[idEs]
        eVec = eVec[:,idEs]
        eVec = np.matmul(aConv, eVec)

        self.logEigVal(eVal)
        self.eigVal = eVal
        self.eigVec = eVec/np.linalg.norm(eVec)


    def updateEquilibrium(self, nnm):
        
        # NNM trained model
        model = nnm.model
        m = nnm.nControlInputs
        n = nnm.nStates
        f = model.predict

        # Number of relevant variables from sparsity procedure
        mask, ids = nnm.sparsityLayer.getSparseMap(opt='states')
        nr = int(np.sum(mask))
        
        # Initial guess
        guess0 = self.equilibrium
        u0 = [0.0 for _ in range(m)]
        u0 = np.array(u0)

        its = [guess0]
        nit = self.nNewtonSteps

        sess = nnm.sess

        stateMatrix = self.stateMatrixReduced

        printStr = "    Computing equilibrium ({}/{})"

        # Estimate equilibrium through Newton method
        for i in range(nit):
            print(printStr.format(i+1,nit), end='\r')

            # Compute A*x for the current equilibrium 
            curGuess = its[-1]
            v0 = np.concatenate((u0,curGuess), axis=0)
            
            # Compute linearization throuh backpropagation
            stateMatrixVal = sess.run(stateMatrix, feed_dict = {model.input:[v0]})[0,:,:]
            

            # Evaluate f(x) and guess affine term 'b'
            fGuess = f(v0.reshape([1,m+n]))[0]
            b = -np.matmul(stateMatrixVal,curGuess[ids]) + fGuess[ids]

            # x_next = inv(I-A)*b
            Id = np.identity(nr)
            updatedGuess = np.matmul(np.linalg.inv(Id-stateMatrixVal),b)

            # How to expand updated guess? Simply fill with zeros since the values don't matter
            updatedGuessFull = np.zeros(n)
            updatedGuessFull[ids] = updatedGuess

            its.append(updatedGuessFull)

        # Store linearization through regression
        self.stateMatrixReducedVal, eqFix = self.stateMatrixReducedReg(v0)
        self.stateMatrixConvVal = self.stateMatrixConvReg(v0)

        eqFix = np.matmul(self.stateMatrixConvVal, eqFix)


        # Expand equilibrium point at non-relevant variables
        xeq = []
        for it in its:
            v0 = np.concatenate((u0, it), axis=0)
            xeq.append(f(v0.reshape([1, m+n]))[0])

        # Log estimates along iterations (the initial guess is skipped)
        self.logEquilibrium(xeq[1:])
        self.logEqFix(eqFix)

        # The equilibrium point is the final guess
        self.equilibrium = np.array(xeq[-1])

        print('')

    def plotEquilibrium(self, plt, block=True, f=None):
        import matplotlib.pyplot as plt

        cases = self.equilibriumLog
        iniPositions = [0]
        for case in cases:
            iniPositions.append(iniPositions[-1] + len(case))

        iniPositions = iniPositions[1:]

        log = np.concatenate(cases, axis=0)
        plt.figure(f)

        plt.ylabel('Estimated states')
        for x in iniPositions:
            plt.axvline(x=x-1, color='k', linewidth = 0.5)
        plt.plot(log)
        plt.xlabel('Iteration')

        plt.show(block=block)

    
    def plotEquilibriumFixed(self, plt, block=True, f=None):
        import matplotlib.pyplot as plt

        cases = self.equilibriumLog
        iniPositions = [0]
        casesFixed = []

        for i in range(len(cases)):
            iniPositions.append(iniPositions[-1] + len(cases[i]))

            if i>0:
                eqFix = self.eqFixLog[i-1]
                npCase = np.array(cases[i][-1])
                npCase = npCase + eqFix
                casesFixed.append([npCase])

            else:
                casesFixed.append(cases[i])
            

        log = np.concatenate(casesFixed, axis=0)

        plt.figure(f)
        plt.ylabel('Estimated states')
        plt.plot(log)
        plt.xlabel('Iteration')

        plt.show(block=block)

    def plotEigVals(self, plt, block=True, f=None):
        import matplotlib.pyplot as plt


        plt.figure(f)

        plt.plot(self.eigValLog)

        plt.ylabel('Estimated eigenvalues')
        plt.xlabel('Iteration')

        plt.show(block=block)

    def saveLinearizations(self, nnm):
        model = nnm.model
        sess = nnm.sess
        m = nnm.nControlInputs
        u0 = [0.0 for _ in range(m)]
        v0 = np.concatenate((u0,self.equilibrium), axis=0)
        
        aConv = sess.run(self.stateMatrixConv, feed_dict = {model.input:[v0]})[0,:,:]
        aRed = sess.run(self.stateMatrixReduced, feed_dict = {model.input:[v0]})[0,:,:]
        bConv = sess.run(self.controlMatrixConv, feed_dict = {model.input:[v0]})[0,:,:]
        bRed = sess.run(self.controlMatrixReduced, feed_dict = {model.input:[v0]})[0,:,:]
        
        np.savetxt('lin_data/aConv.txt', aConv)
        np.savetxt('lin_data/aRed.txt', aRed)
        np.savetxt('lin_data/bConv.txt', bConv)
        np.savetxt('lin_data/bRed.txt', bRed)