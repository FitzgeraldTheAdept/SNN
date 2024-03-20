# This file contains the neuron class
# Adapted From: Izhikevich, Eugene M. "Simple model of spiking neurons." 
# IEEE Transactions on neural networks 14.6 (2003): 1569-1572

from math import pow
from synapse import Synapse

class Neuron(object):
    #connects = []    # Neuron connections
    #pot = []            # membrane potential
    params = {}
    # Regular Spiking Params
    params['a'] = 0.02
    params['b'] = 0.2
    params['c'] = -65
    params['d'] = 8
    
    def __init__(self):
        self.v = []
        self.v[0] = self.params['c'] # membrane potential in millivolts
        self.inSyns = [] # input synapses
        self.outSyns = [] # output synapses
        self.u = self.params['b'] * self.v

        return self


    def _calcI(self):
        # calculates the input current I based on the synapses
        return 10
    
    def step(self, t : int, dt : float):
        # time step for the neuron, update the model
        # inputs: 
        #       t = time index
        #      dt = time step (in ms)

        I = self._calcI(self) # Calculates injected current 
        vnow = self.v[t] # current membrane potential
        dv = (0.04 * pow(vnow,2) + 5 * vnow + 140 - self.u + I) * dt
        du = (self.params['a'] * (self.params['b']*vnow - self.u)) * dt

        # Adjust the variables
        self.v.append(vnow + dv)
        self.u = self.u + du

        # Reset if needed
        if self.v[-1] >= 30:
            self.v[-1] = self.params['c']
            self.u = self.u + self.params['d']

        pass
        

    def connect(self, synapse : Synapse, IO : int):
        # registers the synapse as a connection with this neuron
        # inputs:
        #       synapse = synapse object to connect with
        #       IO      = specifies input or output, 1 for input and 0 for output

        pass

