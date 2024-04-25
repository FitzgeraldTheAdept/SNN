# This file contains the neuron class
# Adapted From: Izhikevich, Eugene M. "Simple model of spiking neurons." 
# IEEE Transactions on neural networks 14.6 (2003): 1569-1572

from math import pow
import random
import funcs
from synapse import Synapse

_INPUT = 1
_OUTPUT = 0
_HIDDEN = 2
_PAIN = -1

class Neuron(object):
    
    #connects = []    # Neuron connections
    #pot = []            # membrane potential
    params = {}
    # Regular Spiking Params
    params['a'] = 0.02
    params['b'] = 0.2
    params['c'] = -65
    params['d'] = 8
    
    def __init__(self, type : int, mxI : float = 80):
        self.v = list()
        #self.v = list()
        self.v.append(self.params['c']) # membrane potential in millivolts
        self.inSyns = list()             # input synapses
        self.outSyns = list()            # output synapses
        self.u = self.params['b'] * self.v[0]
        self.type = type                # 0 = output, 1 = input, 2 = hidden, -1 = pain
        self.I = list()

        self.mxI = mxI                     # maximum synapse current

        self.spikes = list()            # list of times when a spike occurred

        

    """     Private Functions   """
    def _calcI(self, simStep : int, dt : float = 0.1):
        """
            Calculates the input current I based on the synapses.  Weighting happens in the synapses.
            This will compute the sum of all the weighted inputs from the synapses
        
            Inputs:
                simStep - simulation time index (timestamp)
                dt      - time step in (ms)
        """
        totalI = 0
        for syn in self.inSyns:
            totalI = totalI + syn.step(simStep = simStep)
        
        return totalI
    
    """     Public Functions    """
    def step(self, simStep : int, dt : float, I_in : float = 0):
        """
            Time step for the neuron, update model variables
            Inputs: 
                simStep = simulation time index (integer)
                dt      = time step (in ms)
                I_in    = input current
        """
        
        if self.type is _INPUT:
            I = I_in
        else:
            I = self._calcI(simStep = simStep, dt = dt) # Calculates injected current 

        if self.type is _PAIN:
            I = I + I_in

        # saturation check
        if I > self.mxI:
            I = self.mxI

        elif I < -self.mxI:
            I = -self.mxI

        vnow = self.v[simStep] # current membrane potential
        dv = (0.04 * pow(vnow,2) + 5 * vnow + 140 - self.u + I) * dt
        du = (self.params['a'] * (self.params['b']*vnow - self.u)) * dt


        # Adjust the variables
        self.v.append(vnow + dv)
        self.u = self.u + du

        # Reset if needed
        if self.v[-1] >= 30:
            self.v[-1] = self.params['c']
            self.u = self.u + self.params['d']
            if len(self.spikes) == 0:
                self.spikes.append(simStep)
            
            elif self.spikes[-1] < simStep - 4 / dt:
                # Count only the first spike.  This is to combat model instability
                self.spikes.append(simStep)

        # normalized current
        self.I.append(I/self.mxI)

        

    def regSynapse(self, syn, IO : int):
        """
        Registers a new synaptic connection as either an input (1) or output (0) synapse connection to this neuron.
        Adds to self.inSyns or self.outSyns

        Inputs:
            syn - Synapse to be registered
            IO -  neuron is postsynaptic/ synapse is an input (1) or presynaptic / synapse is an output (0) 
            
        """

        if IO == 1:
            self.inSyns.append(syn)
        elif IO == 0:
            self.outSyns.append(syn)
        else:
            raise ValueError('Illegal value for IO: must be 1 (if neuron is postsynaptic) or 0 (presynaptic)')


    def connect(self, toNeuron : object, prePost : int, ispike : list, weight : float = -256):
        """
            Registers a connection between this Neuron and another Neuron
            Inputs:
                toNeuron = neuron object to connect with
                prePost  = 0 for this neuron being the presynaptic, 1 for it being the post
                ispike = shape of the current spike
                weight = neuron weight

        """
        
        if prePost == 0:
            # This neuron is the presynaptic
            if weight == -256:
                syn = Synapse(preNeuron=self, postNeuron=toNeuron, weight=(0.75*random.random() + 0.25) * self.mxI, ispike=ispike)
            else:
                syn = Synapse(preNeuron=self, postNeuron=toNeuron, weight=weight, ispike=ispike)

            # Register the synapse connection with pre neuron
            self.regSynapse(syn=syn, IO=0)
            # register with the post neuron
            toNeuron.regSynapse(syn=syn, IO = 1)

        elif prePost == 1:
            # This neuron is the postsynaptic 
            if weight == -256:
                syn = Synapse(preNeuron=toNeuron, postNeuron=self, weight= random.random() * self.mxI, ispike=ispike)
            else:
                syn = Synapse(preNeuron=toNeuron, postNeuron=self, weight=weight, ispike=ispike)

            # Register the synapse connection with pre neuron
            self.regSynapse(syn=syn, IO=1)
            # register with the post neuron
            toNeuron.regSynapse(syn=syn, IO = 0)

        else:
            raise ValueError('Illegal prePost Value: must be 0 for pre- or 1 for post- synaptic')
        
       
    def adjustWeights(self, lr : float, dt : float, pD : int, iDur : int):
        """
            Adjusts the weights of all this neurons output synapses
            INPUTS:
                lr      - learning rate
                dt      - time step
                pD      - phase duration
                iDur    - ignore duration
        """
        
        for syn in self.outSyns:
            syn.adjustWeight(lr, dt, pD, iDur)

    def reset(self):
        """
            Reset after the simulation
        """
        self.spikes = list()
        self.v = list()
        self.I = list()
        self.v.append(self.params['c']) # membrane potential in millivolts
        self.u = self.params['b'] * self.v[0]
        

    def getAct(self, dt : float, pD : int, iDur : int):
        """
            Get the activity for this neuron
            INPUTS:
                dt      - time step
                pD      - phase duration
                iDur    - ignore duration
        """
        #return funcs.actQuant(spikes=self.spikes, 
        #                        dt=dt,
         #                       endTime=pD,
         #                       startTime=iDur)
        return funcs.actQuant2(cur=self.I)



