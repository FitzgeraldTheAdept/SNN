# Contains the network trainer and helper functions
from math import modf
import json
import numpy as np
import random as rng

from dataGen import CROSS as _DGCROSS, HBAR as _DGHBAR, VBAR as _DGVBAR

_TRAIN = 0
_TEST = 1
_VALID = 2

# output mappings (index of output in the list of outputs)
_CROSS = 0
_VBAR = 1
_HBAR = 2

class Environment(object):
    def __init__(self, 
                 net : object, 
                 stablePause : int = 20, 
                 minFullOn : float = 1.0,
                 maxFullOff: float = 0.05):

        self.net = net
        self.stablePause = stablePause # time to wait for network to stabilize after startup
        self.outputs = self.net.neurons[-1]
        self.painIs =  list(np.empty(self.net.structure[1])) # pain currents
        self.minFullOn = minFullOn # min value for activity to be considered "fully on"
        self.maxFullOff = maxFullOff # max value for activity to be consider "fully off"
        
        self.truth = int()   # indicating which output neuron should be on
        
           

    def setTruth(self, outOn : int) -> None:
        """
            Update which output is designated as the correct one to be active
            INPUTS:
                outOn - an integer index specifying which output in numerical order should be most active
        """
        if outOn >= len(self.outputs):
            e = Exception(f"Selected correct Neuron (output {outOn}) does not exist!")
            raise e
        else:
            self.truth = outOn
        

    def evalOutput(self, minFullOn : float = None, maxFullOff : float = None) -> list:
        """
            Evaluates the output of the network, and generates appropriate pain neuron currents
            INPUTS:
                - Network output voltage spiking activity
                - truth
            OUTPUTS:
                - Currents for the pain neurons
        """

        
        if minFullOn is None:
            minFullOn = self.minFullOn

        if maxFullOff is None:
            maxFullOff = self.maxFullOff
        
        
        # Find spikes for each output
        #spikes = list(map(lambda x : x.spikes, self.outputs))
        # find the current for each output
        cur = list(map(lambda x : x.I, self.outputs))

        # Evaluate activity of each neuron
        #activity =list(map(fn, spikes))
        #activity = list(map(fn, cur))
        activity = self.net.getOuts()

        ###### Evaluate against the truth ######
        self.ppCurrents(activity=activity)
        
        return self.painIs
        """
        # go through each neuron
        for i in range(0, len(self.outputs), 1):
            if i == self.truth and activity[i] < 0.6:
                # this output neuron should be active, but it isn't
                # turn off the pain neuron
                #self.painIs[i] = 0.6 - activity[i] # DEBUG: had minFullOn 
                self.painIs[i] = 0
                #if self.painIs[i] < 0:
                #    self.painIs[i] == 0

            elif activity[i] > 0.2:
                # this neuron should NOT be active, but it do be
                self.painIs[i] = activity[i] - 0.2 # DEBUG: had maxFullOff
                #if self.painIs[i] < 0:
                 #   self.painIs[i] =
            else:
                # Neurons are behaving
                self.painIs[i] = 0

        return self.painIs
        """


    def ppCurrents(self, activity : list) -> list:
        """
            Find the currents 
        
        """
        # PainPleasure Neuron behavior now:
        # pp0 = + if all outputs are off, inject current proportional to offness
        #       - if all outputs are too on, inject current proportional to on-ness 
        # pp1 = + reward neuron for correct output activation, higher if by higher margin
        #       - reward 

        desiredOutAct = activity[self.truth]

        maxAct = np.max(activity)
        minAct = np.min(activity)

        numEq = 0
        for i in range(0, len(activity)):
            if activity[i] == maxAct:
                numEq = numEq + 1

        if numEq > 1:
            # multiple signals share the same max activity
            # we want to punish this
            self.painIs[0] = -0.5
        if numEq == 1:
            # only one signal is the strongest
            # reward this regardless of whether it's correct
            self.painIs[0] = 1


        difs = maxAct - np.asarray(activity)
        margin = 1.0
        # find the second smallest
        for i in range(0, len(difs)):
            if activity[i] != maxAct and difs[i] <= margin:
                margin = difs[i]
        
        self.painIs[1] = 0
        # Now, for pp0: are all outputs off?
        if maxAct < self.maxFullOff:
            # all outputs are suppressed.  Excite the network
            self.painIs[1] = 1*(1.0 - minAct)
            self.painIs[0] = self.painIs[1]
            #print("ENV: All outputs suppressed") # DEBUG
        elif minAct > self.minFullOn:
            # all outputs are too excited.  Suppress the network
            self.painIs[1] = 1*(self.minFullOn - minAct)
            self.painIs[0] = self.painIs[1]
            #print(f"ENV: All outputs excited. injecting: {self.painIs[0]}") # DEBUG
            
        if self.painIs[1] > 1:
            self.painIs[1] = 1.0
        elif self.painIs[1] < -1:
            self.painIs[1] = -1.0

        # Only start kicking this neuron in after network is quieted down
        if self.painIs[1] > 0.2 or self.painIs[1] < -0.2:
            self.painIs[2] = 0
        else:
            # for pp1: is the correct output neuron highest?
            if desiredOutAct == maxAct and numEq == 1:
                
                self.painIs[2] = 5*margin
                #print(f"ENV: Correct output highest. injecting: {self.painIs[1]}") # DEBUG

            else:
                # correct output is not the highest
                
                self.painIs[2] = -5*margin
                #print(f"ENV: Correct Output NOT highest. injecting: {self.painIs[1]}") # DEBUG

        if self.painIs[2] > 1:
            self.painIs[2] = 1.0
        elif self.painIs[2] < -1:
            self.painIs[2] = -1.0
        
        
class Trainer(object):
    def __init__(self, 
                 network : object, 
                 dataPath : str = "./data/", 
                 learnRate : float = 1,
                 numGens : int = 20, # generations between testing epochs
                 numEpochs : int = 10, # number of testing epochs for the test
                 backupEpochs : int = 3, # number of epochs after which to save a copy of the network as is
                 resPath : str = "trained" # name of trained network file saved at the end
                 ):
        self.net = network      # neural network to train

        self.env = Environment(net=network, minFullOn = 0.9, maxFullOff=0.7)  # Environment to use for training

        self.dataPath = dataPath # path to training data
        self.lr = learnRate
        self.numGens = numGens
        self.numEpochs = numEpochs
        self.backupEpochs = backupEpochs
        
        # data - no access to the validation
        self.train = list()
        self.test = list()
        self.dataNums = [_TRAIN, _TEST] # number of images for each set (train, test)
        self.error = list() # list storing average network error for each epoch
        self.resPath = resPath

        self._fetchData()

    def trainNetwork(self):
        """
            Actually train the network
                - apply input signals
                - propagate the network (run a phase)
                - Check output against environment
                - Calculate the pain currents from environment output
                - apply the input signals and pain signals from environment
                - Propagate the network again (run a phase)
                - Have the network adjust weights
                - Repeat for numGens
                - save the network weights to a file every backupGens Generations
                - Also save the network weights a file at the end

            INPUTS:
                numGens - number of generations to test against
        """
        i = 0
        for ep in range(0, self.numEpochs):
            for gen in range(0, self.numGens):
                self._generation()
                print(f"Finished Epoch {ep} generation {gen}")
            
            print(f"Epoch {ep} finished.")
            self._test()
            print(f"MSE = {self.error[-1]}")

            i = i + 1 # mark the epochs completed since last backup
            if i >= self.backupEpochs:
                print("Saving a backup.")
                self.net.writeNetwork(path="backup.net")
                i = 0
            
        print("Training finished!")
        self.net.writeNetwork(self.resPath + ".net")
        

        
    
    def _test(self):
        """
            Tests network and evaluates error
        """

        mse = 0
        # evaluate 4 outputs, find mean of mean squared errors
        for i in [0, 1, 2, 3]:
            # fetch a random testing sample
            sample = self.test[int(rng.random() * self.dataNums[_TEST-1])]
            img = list(np.asarray(sample[0]) / 100)
            truthType = self._mapTruth(truthType=sample[1])

            # apply input signal, propagation
            self.net.phase(I_in=img)
            
            outs = np.asarray(self.net.getOuts())
            # determine 'ground truth'
            gt = np.empty(len(outs))
            gt[truthType] = 1
            # calculate the cumulative square error across all outputs
            mse = mse + np.sum(np.square(gt - outs))/len(outs)
        
        mse = mse / 4
            
        self.error.append(mse)

    def _generation(self):
        """
            Runs one generation on the network
        """
        # fetch a random training sample
        sample = self.train[int(rng.random() * self.dataNums[_TRAIN-1])]
        img = list(np.asarray(sample[0]) / 100)
        truthType = self._mapTruth(truthType=sample[1])
        self.env.setTruth(truthType)
        print(f"TRAINER: Applying with a {truthType} ({img})") #DEBUG

        # apply input signal, initial propagation
        self.net.phase(I_in=img)
        
        #print(f"TRAINER: Outputs are : {self.net.getOuts()}")
        # apply to environment, get the pain currents
        pain_Is = self.env.evalOutput()
        #print(f"TRAINER: Pain Currents are: {pain_Is}") # DEBUG
        
        out_Is = [0.0, 0.0, 0.0]
        out_Is[truthType] = 1.0

        # Repropagate with pain neurons
        self.net.phase(I_in=img, I_pain = pain_Is, I_out = out_Is)

        # Adjust weights
        self.net.adjustWeights(lr = self.lr)

        
    def _getImg(self, whichSet : int) -> list:
        """
            Private method. 
            Gets a random image (with correponding truth label) from the selected data list
            INPUTS:
                whichSet : an integer specifying data set to pull from: 1 
        """
        rng.seed() # Seeds with system random variable seed generator
        imgInd = int(rng.random() * self.dataNums[whichSet])
        # go fetch that image out of the dataset
        
        if whichSet == _TRAIN:
            imgDat = self.train[imgInd] 
        elif whichSet == _TEST:
            imgDat = self.test[imgInd]
        else:
            e = Exception(f"{whichSet} Not a valid data set index for trainer.  Must be 0 or 1 (train or test).")
            raise e

        # training data has both ground truth value and image
        # extract just the image part
        img = imgDat[0]
        # divide by 100, as pixel brightness generated as an int 0-100
        img = list(np.asarray(img) / 100)

        # Retrieve the image truth value (what kind is it). Convert
        imgTruth = self._mapTruth(truthType = imgDat[1])
        
        # return the image itself and the truth value in a list
        return [img, imgTruth]

        
    def _mapTruth(self, truthType : int)-> int:
        """
            Private method.
            maps the truth value from generated data (i.e. type of image, Cross, Vertical Bar, Horizontal Bar)
                to corresponding output in network
            
            Output 0 = Cross
            Output 1 = Vertical Bar
            Output 2 = Horizontal Bar
        
        """
        if truthType == _DGCROSS:
            return _CROSS
        elif truthType == _DGHBAR:
            return _HBAR
        elif truthType == _DGVBAR:
            return _VBAR
        else:
            e = Exception(f"Image type ({truthType}) Not recognized.")
            raise e

    def _fetchData(self):
        """
            Private method.
            Retrieves testing, training, and validation data from specified path, loads into memory
        """
        # Open the test data file
        try:
            with open('{}{}.json'.format(self.dataPath,"test"), 'r') as f:
                data = json.load(f)
                self.test = data['Images']
                self.dataNums[0] = data['Num Images']
        except Exception as e:
            raise e
        
        # Open the training data file
        try:
            with open('{}{}.json'.format(self.dataPath,"train"), 'r') as f:
                data = json.load(f)
                self.train = data['Images']
                self.dataNums[1] = data['Num Images']
        except Exception as e:
            raise e
        
        """ Trainer doesn't have access to validation data
        # Open the validation data file
        try:
            with open('{}{}.json'.format(self.dataPath, "valid"), 'r') as f:
                data = json.load(f)
                self.valid = data['Images']
                self.dataNums[2] = data['Num Images']
        except Exception as e:
            raise e
        """
        

