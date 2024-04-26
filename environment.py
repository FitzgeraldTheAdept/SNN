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
        #spikes = list(map(lambda x : x.spikes, self.outputs))
        # find the current for each output
        cur = list(map(lambda x : x.I, self.outputs))
        
        """
        fn = lambda x : funcs.actQuant(spikes = x, 
                                    dt=self.net.dt, 
                                    endTime=self.net.phaseDuration,
                                    startTime=self.stablePause)
                                    """
        fn = lambda x : funcs.actQuant2(cur=x)
        
        # Evaluate activity of each neuron
        #activity =list(map(fn, spikes))
        activity = list(map(fn, cur))

        ###### Evaluate against the truth ######
        self.ppCurrents(activity=activity)
        
        return self.painIs
        """
        # go through each neuron
        for i in range(0, len(self.outputs), 1):
            if i == self.truth and activity[i] < 0.6:
                # this output neuron should be active, but it isn't
                # turn off the pain neuron
                #self.painIs[i] = 0.6 - activity[i] # DEBUG: had minFullOn 
                self.painIs[i] = 0
                #if self.painIs[i] < 0:
                #    self.painIs[i] == 0

            elif activity[i] > 0.2:
                # this neuron should NOT be active, but it do be
                self.painIs[i] = activity[i] - 0.2 # DEBUG: had maxFullOff
                #if self.painIs[i] < 0:
                 #   self.painIs[i] =
            else:
                # Neurons are behaving
                self.painIs[i] = 0

        return self.painIs
        """


    def ppCurrents(self, activity : list) -> list:
        """
            Find the currents 
        
        """
        # PainPleasure Neuron behavior now:
        # pp0 = + if all outputs are off, inject current proportional to offness
        #       - if all outputs are too on, inject current proportional to on-ness 
        # pp1 = + reward neuron for correct output activation, higher if by higher margin
        #       - reward 

        desiredOutAct = activity[self.truth]

        maxAct = np.max(activity)
        minAct = np.min(activity)

        difs = maxAct - np.asarray(activity)
        margin = 1.0
        # find the second smallest
        for i in range(0, len(difs)):
            if activity[i] != maxAct and difs[i] <= margin:
                margin = difs[i]
        
        self.painIs[0] = 0
        # Now, for pp0: are all outputs off?
        if maxAct < self.maxFullOff:
            # all outputs are suppressed.  Excite the network
            self.painIs[0] = 2*(1.0 - minAct)
            #print("ENV: All outputs suppressed") # DEBUG
        elif minAct > self.minFullOn:
            # all outputs are too excited.  Suppress the network
            self.painIs[0] = 20*(self.minFullOn - minAct)
            #print(f"ENV: All outputs excited. injecting: {self.painIs[0]}") # DEBUG
            
        if self.painIs[0] > 1:
            self.painIs[0] = 1.0
        elif self.painIs[0] < -1:
            self.painIs[0] = -1.0

        # for pp1: is the correct output neuron highest?
        if desiredOutAct < maxAct + .001 and desiredOutAct > maxAct - .001:
            
            self.painIs[1] = 50*margin
            #print(f"ENV: Correct output highest. injecting: {self.painIs[1]}") # DEBUG

        else:
            # correct output is not the highest
            
            self.painIs[1] = -50*margin
            #print(f"ENV: Correct Output NOT highest. injecting: {self.painIs[1]}") # DEBUG

        if self.painIs[1] > 1:
            self.painIs[1] = 1.0
        elif self.painIs[1] < -1:
            self.painIs[1] = -1.0
        
        