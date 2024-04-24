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
    def __init__(self, 
                 path : str = None,
                 phaseDuration : int = 100, 
                 dt : float = 0.1, 
                 structure : list = [2, 1, 1], 
                 simStep : int = 0):
        if path is None:
            # build a new network
            self.phaseDuration = phaseDuration
            self.dt            = dt
            self.structure     = structure

            self.t             = list(map(lambda x: x * self.dt, range(0, int(self.phaseDuration / self.dt),1)))
            self.simStep       = simStep
            self.neurons       = self.buildNetwork(self.structure)
        else:
            # load a network from file
            self.loadNetwork(path=path)

    
    def loadNetwork(self, path : str):
        """ 
            Load a network from a file
            Intialize all synapses to weights in the file

        """
        pass

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

        # define the spike shape
        ispikeTotal = funcs.ispike(dt=self.dt)
        ispikeshape = ispikeTotal['current']

        # connect all the input neurons to the pain neurons
        self.fillConnects(fromLayer=neurons[0], toLayer=neurons[1], ispike=ispikeshape)
        
        if numHideLays > 0:
            # connect all the input and pain neurons to the first hidden layer, if it exists
            self.fillConnects(fromLayer=neurons[0], toLayer=neurons[2], ispike=ispikeshape)

            # connect pain neurons to all hidden layers
            hlayer = 2 # 0th hidden layer
            while hlayer - 1 < numHideLays: # account for 2 offset
                self.fillConnects(fromLayer=neurons[1], toLayer=neurons[hlayer], ispike=ispikeshape)
                hlayer = hlayer + 1

            # fill in the rest of the layers as normal
            layer = 2 # 0th hidden layer
            while layer - 1 < numHideLays:
                self.fillConnects(fromLayer=neurons[layer], toLayer=neurons[layer + 1], ispike=ispikeshape)
                layer = layer + 1

        return neurons
    
    def fillConnects(self, fromLayer : list, toLayer : list, ispike : list):
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

    def phase(self, I_in : list, I_pain : list):
        """
            Execute a phase
            In other words, solve the whole network for the current simStep, then increment to the next step
            Inputs:
                I_in - currents derived from the input strength, list of floats
                I_pain - currents to the pain neurons

        """
        while self.simStep < len(self.t):
            
            # Solve all the neurons, starting with the input layer and moving forward
            for layer in self.neurons:
                for neu in layer:
                    if neu.type == 1:
                        neu.step(simStep = self.simStep, dt = self.dt, I_in = I_in)
                        
                    elif neu.type == -1:
                        # pain neuron
                        neu.step(simStep = self.simStep, dt = self.dt, I_in = I_pain )
                        # Neurons will check synapses to find their other input current
                    else:
                        # other neuron
                        neu.step(simStep = self.simStep, dt = self.dt)
                        # Neurons will check synapses to find their input current

            # increment to next simulation step
            self.simStep = self.simStep + 1

    def adjustWeights(self):
        """
            Adjusts the weights of the network using hebbian learning rules in trainer file
        
        """
        pass


if __name__ == '__main__':
    net = Network(structure=[1,2,3,4,5,6])
    net.drawNetwork()
    
