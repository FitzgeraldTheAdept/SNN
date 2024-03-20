# This file contains the synapse class

from neuron import Neuron

class Synapse(object):

    def __init__(self, preNeuron : Neuron, postNeuron : Neuron, weight = float):
        self.pre = preNeuron    # Presynaptic Neuron
        self.post = postNeuron  # Postsynaptic Neuron
        self.weight = weight    # Neuron Weight
        return self
    

    def calcStep(self):
        synI = self.pre.v 