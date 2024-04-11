import numpy as np
import math
import matplotlib.pyplot as plt
import funcs
from neuron import Neuron

class Network(object):
    """
        Overall Network Object

        Fields:
        phaseDuration   - time in ms that each phase (prop, pain, etc.) lasts
        dt              - simulation time step dt (in ms)
        structure       - list containing number of neurons in each layer of the network.
                          [inputs, pain neurons, outputs, hidden layer 1, hidden layer 2,...]
        t               - time vector for each phase
        simStep         - Simulation step (which time index in vector t are we)
        neurons         - 2D list of Neuron objects [inputs, pain, hl1, hl2, ..., output]
    """
    def __init__(self, phaseDuration : int = 100, dt : float = 0.1, structure : list = [2, 1, 1], simStep : int = 0):
        self.phaseDuration = phaseDuration
        self.dt            = dt
        self.structure     = structure


        self.t             = list(map(lambda x: x * self.dt, range(0, int(self.phaseDuration / self.dt),1)))
        self.simStep       = simStep
        self.neurons       = self.buildNetwork(self.structure)

    
    def buildNetwork(self, structure : list):
        """
            Builds the network based on the structure
            Inputs:
                structure   - list containing number of neurons in each layer of the network.
                              [inputs, pain neurons, outputs, hidden layer 1, hidden layer 2,...]
            Outputs:
                neurons     - list of Neurons, according to shape of structure
        """
        # Extract dimensions for the network for readability
        numIns      = structure[0]
        numPains    = structure[1]
        numOuts     = structure[2]
        # find the number of hidden layers
        numHideLays = len(structure) - 3

        Ins     = [Neuron(type=1)] * numIns     # 1 = input neuron
        Pains   = [Neuron(type=-1)] * numPains  # -1 = pain neuron
        Outs    = [Neuron(type=0)] * numOuts    # 0 = output neuron    

        neurons = [Ins, Pains]
        
        # Add in hidden layers
        i = 3
        while i < len(structure):
            neurons.append([Neuron(type=2)] * structure[i]) # 2 = hidden neuron
            i = i + 1
                
        # add in the outputs as the last layer
        neurons.append(Outs)

        # connect all the input neurons to the pain neurons
        self.fillConnects(neurons[0], neurons[1])
        
        if numHideLays > 0:
            # connect all the input and pain neurons to the first hidden layer, if it exists
            self.fillConnects(neurons[0], neurons[2])
            self.fillConnects(neurons[1], neurons[2])

            # fill in the rest of the layers
            layer = 2 # 0th hidden layer
            while layer - 1 < numHideLays:
                self.fillConnects(neurons[layer], neurons[layer + 1])
                layer = layer + 1

        return neurons
    
    def fillConnects(self, fromLayer : list, toLayer : list):
        """
            initializes all connections from neurons in fromLayer to neurons in toLayer

            Inputs:
                fromLayer   - list of neurons in presynaptic layer
                toLayer     - list of neurons in postsynaptic layer
            
            Outputs:
                None
        """

        # Connect the layers
        for fromNeu in fromLayer:
            for toNeu in toLayer:
                # fromNeu is the presynaptic connection of the toNeu
                fromNeu.connect(toNeu, 0)
        
    
    def drawNetwork(self):
        """
            Draws the Network diagram
        """
        for i in self.neurons:
            print(len(i))

    def step(self, I_in : list):
        """
            Advance the network 1 step in the simulation.
            In other words, solve the whole network for the current simStep, then increment to the next step
            Inputs:
                I_in - currents derived from the input strength, list of flaots

        """
        while self.simStep < len(self.t):
            
            # Solve all the neurons, starting with the input layer and moving forward
            for layer in self.neurons:
                for neu in layer:
                    neu.step(simStep = self.simStep, dt = self.dt )
                    # Neurons will call synapses to find their current


            # increment to next simulation step
            self.simStep = self.simStep + 1
        
            
        


def simTick():
    """
        "Tick" the simulation one time step
    """
    pass


if __name__ == '__main__':
    net = Network(structure=[1,2,3,4,5,6])
    net.drawNetwork()
    
