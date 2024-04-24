# Contains the network trainer and helper functions
import funcs
import json
import numpy as np
import random as rng
from network import Network
from environment import Environment
from neuron import Neuron
from synapse import Synapse

from dataGen import CROSS as _DGCROSS, HBAR as _DGHBAR, VBAR as _DGVBAR

_TRAIN = 0
_TEST = 1
_VALID = 2

# output mappings (index of output in the list of outputs)
_CROSS = 0
_VBAR = 1
_HBAR = 2


def hebbian1(syn : Synapse, dt : float = 0.01, phaseDur : int = 300, ignoreDur : int = 20) -> float:
    """
        Calculates a "Hebbian Coefficient", which ranges from -1.0 to 1.0 and quantifies how much
        the synapse weight should be adjusted based on the correlation between the two neurons
        
        This uses both the correlation score, which quantifies causality from pre-> post, and the 
        differences in rates between neuron 1 and neuron 2, to derive the Hebbian Coefficient

        INPUTS:
            - Synapse

    """

    preSpikes = syn.pre.spikes
    postSpikes = syn.post.spikes
    # Calculate the correlation between these two spiking signals
    corr = funcs.actCompare(spikes1=preSpikes,
                     spikes2=postSpikes,
                     dt=dt,
                     endTime=phaseDur,
                     startTime=ignoreDur,
                     maxDelay=35)
    
    # Also calculate the activity of each neuron separately
    act1 = funcs.actQuant(spikes=preSpikes,
                        dt = dt, 
                        endTime= phaseDur, 
                        startTime = ignoreDur)
    
    act2 = funcs.actQuant(spikes=postSpikes,
                        dt = dt, 
                        endTime= phaseDur, 
                        startTime = ignoreDur)
    
    maxAct = (phaseDur - ignoreDur)/4 # 4 ms

    # This is where hands are waved
    # See if either neurons was active enough
    if act1 > 0.02 or act2 > 0.02:
        actDif = act1 - act2 # activity difference

        if actDif < 0 and corr > 0.5:
            # Neuron 2 is more active than neuron 1
            # BUT they have high correlation
            # POSITIVE HEBBIAN COEFFICIENT
            # proportional to how strongly correlated they are
            # inversely proportional to activity of neuron 2 ?????
            hebCoef = corr # / act2

        elif actDif < 0 and corr <= 0.5:
            # Neuron 2 is more active than neuron 1 
            # AND they have low correlation
            # NEGATIVE HEBBIAN COEFFICIENT
            #hebCoef = actDif
            hebCoef = corr - 1

        elif actDif > 0 and corr > 0.5:
            # Neuron 1 is more active than neuron 2
            # BUT they have a high correlation score
            # POSITIVE HEBBIAN COEFFICIENT
            hebCoef = corr # / act1

        elif actDif > 0 and corr <= 0.5:
            # Neuron 1 is more active than neuron 2
            # AND they have a low correlation score
            # NEGATIVE HEBBIAN COEFFICIENT
            #hebCoef = -1 * actDif
            hebCoef = corr - 1
        
        return hebCoef

    else:
        return 0.0
    
class Trainer(object):
    def __init__(self, network : Network, environment : Environment, dataPath : str = "./data/" ):
        self.net = network      # neural network to trian
        self.env = environment  # Environment to use for training
        self.dataPath = dataPath # path to training data
        
        # data - no access to the validation
        self.train = list()
        self.test = list()
        self.dataNums = [_TRAIN, _TEST] # number of images for each set (train, test)
        self.error = list() # list storing average network error for each generation

        self._fetchData()

    def trainNetwork(self, numGens : int = 100, backupGens : int = 20):
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
        pass

    def adjustWeights(self):
        """
            Adjust the weights of the network using the hebbian learning rules 
            - Might be moved inside the network object
        """
        pass

        
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
        

