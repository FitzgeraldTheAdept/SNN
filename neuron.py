# This file contains the neuron class
# Adapted From: Izhikevich, Eugene M. "Simple model of spiking neurons." 
# IEEE Transactions on neural networks 14.6 (2003): 1569-1572

# Model changed to a simple, linear integrate and fire model:
# Using Q=CV, setting C = 1, so a direct mapping of Q to V
# In this implementation, Q loses any actual meaning, other than being Coulombs/Farad
# I is calculated using the same spike-and-sum as original neuron model
# Q is the sum of I(t-n)*dt, from n = length(IspikeShape) to 0
#
# using nominal voltages
# If q is below -65, there is a passive regen of +0.5/dt, in addition to current TODO: make these dependent on the charge difference (exponential)
# If q is above -65, there is a passive change of -0.5/dt, in addition to current
# If q is within 0.5 of -65 and injected current is 0, q = -65
# If q exceeds +30, a spike is recorded, and q is reset to -80

from math import pow
import random
import funcs
from synapse import Synapse

_INPUT = 1
_OUTPUT = 0
_HIDDEN = 2
_PAIN = -1

class Neuron(object):
    
    model = {}
    # Model parameters
    model['reset'] = -80 # reset value
    model['thresh'] = 30 # threshold value to reset at
    model['rest'] = -65 # resting voltage/charge
    model['pass_loss'] = (model['thresh'] - model['reset'])/4 # passive loss when above rest value
    model['pass_gain'] = model['pass_loss'] # passive gain when below rest value
    
    def __init__(self, type : int, mxI : float = 27.5):
        self.v = list()
        self.v.append(self.model['rest'])   # nominal membrane potential in millivolts
        self.inSyns = list()                # input synapses
        self.outSyns = list()               # output synapses
        self.type = type                    # 0 = output, 1 = input, 2 = hidden, -1 = pain
        self.I = list()                     # input current
        self.I.append(0)                    # initial input current is 0
        self.dir = 1                        # 1 if positive output, -1 if negative output (peripheral neurons only)

        self.mxI = mxI                     # maximum synapse current

        self.spikes = list()            # list of times when a spike occurred

        

    """     Private Functions   """
    def _calcI(self, simStep : int):
        """
            Calculates the input current I based on the synapses.  Weighting happens in the synapses.
            This will compute the sum of all the weighted inputs from the synapses
        
            Inputs:
                simStep - simulation time index (timestamp)
        """
        totalI = 0
        for syn in self.inSyns:
            if syn.pre.type ==_PAIN:
                
                totalI = totalI + syn.pre.dir * syn.step(simStep = simStep)
            else:
                totalI = totalI + syn.step(simStep = simStep)
        
        return totalI
    
    """     Public Functions    """
    def step(self, simStep : int, dt : float, I_in : float = None):
        """
            Time step for the neuron, update model variables
            Inputs: 
                simStep = simulation time index (integer)
                dt      = time step (in ms)
                I_in    = input current
        """
        
        if self.type == _INPUT:
            I = I_in*self.mxI

        elif I_in is None and self.type == _PAIN:
            I = 0

        elif self.type == _PAIN:
            if I_in < 0:
                self.dir = -1
                
            else:
                self.dir = 1
                
            I = I_in * self.dir * self.mxI

        elif self.type == _OUTPUT and I_in is not None:
            # training
            I = I_in * self.mxI
            
        else:
            I = self._calcI(simStep = simStep) # Calculates injected current 
            #print("Here2")

        # saturation check
        if I > self.mxI:
            I = self.mxI

        elif I < -self.mxI:
            I = -self.mxI

        # change in "charge"
        dq = dt * I

        vnow = self.v[simStep] # current membrane potential

        if vnow >= self.model['thresh']:
            # need to reset
            self.v.append(self.model['reset'])
            # record spike time
            self.spikes.append(simStep)
            
        elif vnow < (self.model['rest'] - self.model['pass_gain'] * dt):
            # include passive gain
            self.v.append(vnow + dq + self.model['pass_gain'] * dt)
        elif vnow > (self.model['rest'] + self.model['pass_gain'] * dt):
            # include passive loss
            self.v.append(vnow + dq - self.model['pass_loss'] * dt)
            
        elif dq < dt*self.mxI*0.01:
            # vnow is practically v_rest, and change in current is minimal
            self.v.append(self.model['rest'])
        else: 
            # No passive gain or loss
            self.v.append(vnow + dq)

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


    def connect(self, toNeuron : object, prePost : int, ispike : list, weight : float = -99999):
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
            if self.type == _PAIN:
                syn = Synapse(preNeuron=self, postNeuron=toNeuron, weight= 1.0 * self.mxI, ispike=ispike)
            elif weight == -99999:
                syn = Synapse(preNeuron=self, postNeuron=toNeuron, weight=(0.5*random.random() + 0.25) * self.mxI, ispike=ispike)
            else:
                syn = Synapse(preNeuron=self, postNeuron=toNeuron, weight=weight, ispike=ispike)

            # Register the synapse connection with pre neuron
            self.regSynapse(syn=syn, IO = 0)
            # register with the post neuron
            toNeuron.regSynapse(syn=syn, IO = 1)

        elif prePost == 1:
            if self.type == _PAIN:
                syn = Synapse(preNeuron=self, postNeuron=toNeuron, weight= 1.0 * self.mxI, ispike=ispike)
            elif weight == -99999:
                syn = Synapse(preNeuron=self, postNeuron=toNeuron, weight=(0.5*random.random() + 0.25) * self.mxI, ispike=ispike)
            else:
                syn = Synapse(preNeuron=self, postNeuron=toNeuron, weight=weight, ispike=ispike)

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
        self.v.append(self.model['rest']) # membrane potential in millivolts
        self.I.append(0)
        

    def getAct(self, dt : float, pD : int, iDur : int):
        """
            Get the activity for this neuron
            INPUTS:
                dt      - time step
                pD      - phase duration
                iDur    - ignore duration
        """
        return funcs.actQuant(spikes=self.spikes, 
                              cur=self.I,
                                dt=dt,
                                endTime=pD,
                                startTime=iDur)
        #return funcs.actQuant2(cur=self.I)



