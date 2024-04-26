# This file contains the synapse class

#maxI = 80   # maximum synapse current.  This gives a refractory period of ~4 ms
maxI = 20
import numpy as np
# from neuron import Neuron
import funcs


def hebbian1(syn : object, dt : float = 0.01, phaseDur : int = 300, ignoreDur : int = 20) -> float:
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
    """
    corr = funcs.actCompare(spikes1=preSpikes,
                     spikes2=postSpikes,
                     dt=dt,
                     endTime=phaseDur,
                     startTime=ignoreDur,
                     maxDelay=35)
    """
    corr = funcs.actCompare(cur1 = syn.pre.I,
                     spikes1=preSpikes,
                     spikes2=postSpikes,
                     dt=dt,
                     endTime=phaseDur,
                     startTime=ignoreDur,
                     maxDelay=35)
    
    # Also calculate the activity of each neuron separately
    """
    act1 = funcs.actQuant2(spikes=preSpikes,
                        dt = dt, 
                        endTime= phaseDur, 
                        startTime = ignoreDur)
    
    act2 = funcs.actQuant2(spikes=postSpikes,
                        dt = dt, 
                        endTime= phaseDur, 
                        startTime = ignoreDur)
    """
    act1 = funcs.actQuant2(cur=syn.pre.I)
    act2 = funcs.actQuant2(syn.pre.I)
    
    #maxAct = (phaseDur - ignoreDur)/4 # 4 ms

    # This is where hands are waved
    # See if either neurons was active enough
    if act1 > 0.02 or act2 > 0.02:
        actDif = act1 - act2 # activity difference

        if actDif <= 0 and corr > 0.5:
            # Neuron 2 is more active than neuron 1
            # BUT they have high correlation
            # POSITIVE HEBBIAN COEFFICIENT
            # proportional to how strongly correlated they are
            # inversely proportional to activity of neuron 2 ?????
            hebCoef = corr # / act2

        elif actDif <= 0 and corr <= 0.5:
            # Neuron 2 is more active than neuron 1 
            # AND they have low correlation
            # NEGATIVE HEBBIAN COEFFICIENT
            #hebCoef = actDif
            hebCoef = corr - 1

        elif actDif >= 0 and corr > 0.5:
            # Neuron 1 is more active than neuron 2
            # BUT they have a high correlation score
            # POSITIVE HEBBIAN COEFFICIENT
            hebCoef = corr # / act1

        elif actDif >= 0 and corr <= 0.5:
            # Neuron 1 is more active than neuron 2
            # AND they have a low correlation score
            # NEGATIVE HEBBIAN COEFFICIENT
            #hebCoef = -1 * actDif
            hebCoef = corr - 1
        
        return hebCoef

    else:
        return 0.0

class Synapse(object):
    
    def __init__(self, preNeuron : object, postNeuron : object, weight : float, ispike : np.array = None):
        self.pre = preNeuron        # Presynaptic Neuron
        self.post = postNeuron      # Postsynaptic Neuron
        self.weight = weight        # Synapse Weight
        if ispike is not None:
            self.ispikeShape = ispike   # Shape of Current Spike
    
    
    def setISpike(self, ispike : np.array):
        """
            Sets Current Spike shape of this synapse
        """
        self.ispikeShape = ispike

    def step(self, simStep : int) -> float:
        """
            Calculates the current for this synapse at this simulation step
            Inputs:
                simStep - simulation time index
            Outputs:
                weighted current from this synapse (including negative if pain)
        """
        synI = 0
        if len(self.pre.spikes) != 0:
            # retrieve spikes from the preneuron
            for spike in self.pre.spikes:
                if simStep - spike < len(self.ispikeShape):
                    # Possible overlapping spikes
                    # Find whichever current value in spike is strongest
                    if spike <= simStep and self.ispikeShape[int(simStep-spike)] > synI:
                        synI = self.ispikeShape[int(simStep-spike)]

        # see if the previous neuron is a pain neuron. If it is, current counts as a negative
        #if self.pre.type == -1:
        #    synI = -1* synI 

        return synI * self.weight


    def adjustWeight(self, lr : float, dt : float, pD : int, iDur : int):
        """
            Adjusts the weight of the synapse, for training.
            Inputs:
                lr       - learning Rate
                dt       - time step
                pD       - phase duration
                iDur       - ignore Duration
            
            Outputs:
                The adjusted weight value, in addition to adjusting the synapse weight

            Calculates Strength using hebbian learning rules
            strength - correlation value telling how strongly to increase (positive) or decrease (negative) the weight
        """
        strength = hebbian1(syn=self, dt=dt, phaseDur=pD, ignoreDur=iDur )

        self.weight = self.weight + lr * strength

        # Handle weight saturation
        if self.weight > maxI:
            self.weight = float(maxI)
        if self.weight < -1*maxI: 
            self.weight = float(-1*maxI)

        return self.weight
    