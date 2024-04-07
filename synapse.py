# This file contains the synapse class
from neuron import Neuron
import numpy as np

class Synapse(object):
    
    def __init__(self, preNeuron : Neuron, postNeuron : Neuron, weight = float, ispike : np.array = None):
        self.pre = preNeuron        # Presynaptic Neuron
        self.post = postNeuron      # Postsynaptic Neuron
        self.weight = weight        # Neuron Weight
        self.curIStart = 0          # Time step index of start of the current I spike
        self.newIStart = 0          # Time step index of start of the most recent I spike
        if ispike is not None:
            self.ispikeShape = ispike   # Shape of Current Spike

        # register with the neurons
        self.pre.regSynapse(self,0)
        self.post.regSynapse(self,1)    
        return self
    
    def setISpike(self, ispike : np.array):
        """
            Sets Current Spike shape of this synapse
        """
        self.ispikeShape = ispike
        # return self
    

    def calcStep(self):
        """
            Calculates the current for this synapse at this simulation step
        """
        synI = self.pre.v 
        pass

   

    def _combineSpike(self, simTStep : int):
        """
        Combine current spikes by taking previous spike and current spike current values 
        and "riding" whichever spike is larger.  I.e. look at the start time for those spikes
        and replace the start time with the start time of the spike with the larger magnitude.
        In reality, there would be a combination here, but this is a simplified approach that we
        can tweak later on.
        """
        
        pass