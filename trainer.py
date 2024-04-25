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

#from network import Network as Net

class Trainer(object):
    def __init__(self, 
                 network : object, 
                 dataPath : str = "./data/", 
                 learnRate : float = 0.01,
                 numGens : int = 20, # generations between testing epochs
                 numEpochs : int = 10, # number of testing epochs for the test
                 backupEpochs : int = 3, # number of epochs after which to save a copy of the network as is
                 resPath : str = "trained" # name of trained network file saved at the end
                 ):
        self.net = network      # neural network to train

        from environment import Environment
        self.env = Environment(net=network, minFullOn = 0.8, maxFullOff=0.1)  # Environment to use for training

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
        
        print(f"TRAINER: Outputs are : {self.net.getOuts()}")
        # apply to environment, get the pain currents
        pain_Is = self.env.evalOutput()
        #print(f"TRAINER: Pain Currents are: {pain_Is}") # DEBUG

        # Repropagate with pain neurons
        self.net.phase(I_in=img, I_pain = pain_Is)

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
        

