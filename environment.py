"""
    Contains the Environment functions for the system
    
"""
from network import Network
import funcs
import numpy as np

class Environment(object):
    def __init__(self, 
                 net : object, 
                 stablePause : int = 20, 
                 minFullOn : float = 1.0,
                 maxFullOff: float = 0.05):

        self.net = net
        self.stablePause = stablePause # time to wait for network to stabilize after startup
        self.outputs = self.net.neurons[-1]
        self.painIs =  list(np.empty(self.net.structure[1])) # pain currents
        self.minFullOn = minFullOn # min value for activity to be considered "fully on"
        self.maxFullOff = maxFullOff # max value for activity to be consider "fully off"

        if len(self.painIs) != len(self.outputs):
            e = Exception(f"Inequal number of pain {len(self.painIs)} and output {len(self.outputs)} neurons!")
            raise e
        
        self.truth = int()   # indicating which output neuron should be on
        
           

    def setTruth(self, outOn : int) -> None:
        """
            Update which output is designated as the correct one to be active
            INPUTS:
                outon - an integer index specifying which output in numerical order should be most active
        """
        if outOn >= len(self.outputs):
            e = Exception(f"Selected correct Neuron (output {outOn}) does not exist!")
            raise e
        else:
            self.truth = outOn
        

    def evalOutput(self, minFullOn : float = None, maxFullOff : float = None) -> list:
        """
            Evaluates the output of the network, and generates appropriate pain neuron currents
            INPUTS:
                - Network output voltage spiking activity
                - truth
            OUTPUTS:
                - Currents for the pain neurons
        """
        if minFullOn is None:
            minFullOn = self.minFullOn

        if maxFullOff is None:
            maxFullOff = self.maxFullOff
        
        
        # Find spikes for each output
        spikes = list(map(lambda x : x.spikes, self.outputs))
        
        fn = lambda x : funcs.actQuant(spikes = x, 
                                    dt=self.net.dt, 
                                    endTime=self.net.phaseDuration,
                                    startTime=self.stablePause)
        
        # Evaluate activity of each neuron
        activity =list(map(fn, spikes))

        ###### Evaluate against the truth ######
        # go through each neuron
        for i in range(0, len(self.outputs), 1):
            if i == self.truth:
                # this neuron should be active
                self.painIs[i] = minFullOn - activity[i]

            else:
                # this neuron should NOT be active
                self.painIs[i] = activity[i] - maxFullOff
                

        return self.painIs


