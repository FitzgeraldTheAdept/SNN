# Contains the network trainer and helper functions
import funcs
from neuron import Neuron
from synapse import Synapse

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
    

    
    


