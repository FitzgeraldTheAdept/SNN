# This file contains the synapse class
from neuron import Neuron
import numpy as np

maxI = 80   # maximum synapse current.  This gives a refractory period of ~4 ms

class Synapse(object):
    
    def __init__(self, preNeuron : Neuron, postNeuron : Neuron, weight = float, ispike : np.array = None):
        self.pre = preNeuron        # Presynaptic Neuron
        self.post = postNeuron      # Postsynaptic Neuron
        self.weight = weight        # Synapse Weight
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
        
    

    def step(self, simStep : int):
        """
            Calculates the current for this synapse at this simulation step
            Inputs:
                simStep - simulation time index
            Outputs:
                weighted current from this synapse (including negative if pain)
        """
        synI = 0
        if not len(self.pre.spikes) is 0:
            # retrieve spikes from the preneuron
            for spike in self.pre.spikes:
                if simStep - spike < len(self.ispikeShape):
                    # Possible overlapping spikes
                    # Find whichever current value in spike is strongest
                    if self.ispikeShape[simStep-spike] > synI:
                        synI = self.ispikeShape[simStep-spike]

        # see if the previous neuron is a pain neuron. If it is, current counts as a negative
        if self.pre.type is -1:
            synI = -1 * synI

        return synI * self.weight

   

    def _combineSpike(self, simTStep : int):
        #### DEFUNCT? ####
        """
        Combine current spikes by taking previous spike and current spike current values 
        and "riding" whichever spike is larger.  I.e. look at the start time for those spikes
        and replace the start time with the start time of the spike with the larger magnitude.
        In reality, there would be a combination here, but this is a simplified approach that we
        can tweak later on.
        """
        
        pass