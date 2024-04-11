# This file contains the neuron class
# Adapted From: Izhikevich, Eugene M. "Simple model of spiking neurons." 
# IEEE Transactions on neural networks 14.6 (2003): 1569-1572

from math import pow
import random
import numpy
#from synapse import Synapse
# Would really like to have the above line but for whatever reason we get a circular import

class Neuron(object):
    #connects = []    # Neuron connections
    #pot = []            # membrane potential
    params = {}
    # Regular Spiking Params
    params['a'] = 0.02
    params['b'] = 0.2
    params['c'] = -65
    params['d'] = 8
    
    def __init__(self, type : int):
        self.v = numpy.array()
        #self.v = list()
        self.v[0] = self.params['c']    # membrane potential in millivolts
        self.inSyns = set()             # input synapses
        self.outSyns = set()            # output synapses
        self.u = self.params['b'] * self.v
        self.type = type                # 0 = output, 1 = input, 2 = hidden, -1 = pain

        self.spikes = list()            # list of times when a spike occurred

        return self

    """     Private Functions   """
    def _calcI(self):
        # calculates the input current I based on the synapses
        return 10
    
    """     Public Functions    """
    def step(self, t : int, dt : float):
        """
            Time step for the neuron, update model variables
            Inputs: 
                t = time index (integer)
               dt = time step (in ms)
        """

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
            self.spikes.append(t)

        

    def regSynapse(self, syn, IO : int):
        """
        Registers a new synaptic connection as either an input (1) or output (0) synapse connection to this neuron.
        Adds to self.inSyns or self.outSyns, after checking to see if it's already been registered.

        Inputs:
            syn - Synapse to be registered
            IO -  neuron is postsynaptic/ synapse is an input (1) or presynaptic / synapse is an output (0) 
            
        """

        if IO == 1:
            self.inSyns.add(syn)
        elif IO == 0:
            self.outSyns.add(syn)
        else:
            raise ValueError('Illegal value for IO: must be 1 (if neuron is postsynaptic) or 0 (presynaptic)')


    def connect(self, toNeuron, prePost : int):
        # registers a connection between this Neuron and another Neuron
        # inputs:
        #       toNeuron = neuron object to connect with
        #       prePost  = 0 for this neuron being the presynaptic, 1 for it being the post
        random.seed(42)
        maxWeight = 80 # This gives a refractory period of ~4 ms, which is about the max
        from synapse import Synapse
        if prePost == 0:
            # This neuron is the presynaptic
            syn = Synapse(self, toNeuron, random.random() * maxWeight)

        elif prePost == 1:
            # This neuron is the postsynaptic 
            syn = Synapse(toNeuron, self, random.random() * maxWeight)

        else:
            raise ValueError('Illegal prePost Value: must be 0 for pre- or 1 for post- synaptic')


        return syn

