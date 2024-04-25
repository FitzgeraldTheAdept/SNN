import numpy as np
import math
import random as rng
import matplotlib.pyplot as plt
import funcs
from neuron import Neuron

_INPUT = 1
_OUTPUT = 0
_HIDDEN = 2
_PAIN = -1

def _list2str(inList : list) -> str:
        """
            Converts the given list to a string with spaces after each of the values
            INPUTS:
                inList - list to convert to a single string
        """
        outStr = ""
        for item in inList:
            outStr + str(item) + " "
        
        return outStr

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
        maxI            - maximum current from a single synapse. Also max synapse weight
    """
    def __init__(self, 
                 path : str = None,
                 phaseDuration : int = 100, 
                 dt : float = 0.1, 
                 structure : list = [2, 1, 1], 
                 simStep : int = 0,
                 maxI : float = 80):
        
        if path is None:
            # build a new network
            self.phaseDuration = phaseDuration
            self.dt            = dt
            self.structure     = structure

            self.t             = list(map(lambda x: x * self.dt, range(0, int(self.phaseDuration / self.dt),1)))
            self.simStep       = simStep
            self.neurons       = self.buildNetwork(self.structure)

            # create the current spike shape
            ispikeTotal = funcs.ispike(dt=self.dt)
            self.ispikeshape = ispikeTotal['current']

            self.maxI = maxI
        else:
            # load a network from file
            self.loadNetwork(path=path)

    
    def loadNetwork(self, path : str):
        """ 
            Load a network from a file
            Intialize all synapses to weights in the file
            INPUTS:
                path - path to file

            OUTPUTS:
                This network object is fully initialized

        """
        self.simStep       = 0
        try:
            with open(path, 'r') as f:
                # first line is formatted as:
                # phase_duration dt maxI
                line = f.readline().strip('\n')
                netParams = line.split(' ')
                                
                self.phaseDuration = int(netParams[0])
                self.dt            = float(netParams[1])
                self.maxI          = float(netParams[2])

                self.t             = list(map(lambda x: x * self.dt, range(0, int(self.phaseDuration / self.dt),1)))

                # Read the current spike shape from the second line
                line = f.readline().strip('\n')
                self.ispikeshape = list(map(float, line.split(' ')))

                # Read in the structure from the third line
                line = f.readline().strip('\n')
                self.structure = list(map(int, line.split(' ')))
                
                # Extract dimensions for the network; to ease readability
                numIns      = self.structure[0]
                numPains    = self.structure[1]
                numOuts     = self.structure[2]
                
                # find the number of hidden layers; minus 3 for input, output, and pain layers
                numHideLays = len(self.structure) - 3

                Ins     = [Neuron(type=1)] * numIns     # 1 = input neuron
                Pains   = [Neuron(type=-1)] * numPains  # -1 = pain neuron
                Outs    = [Neuron(type=0)] * numOuts    # 0 = output neuron    

                neurons = [Ins, Pains]
                
                # Add in hidden layers
                i = 3
                while i < len(self.structure):
                    neurons.append([Neuron(type=2)] * self.structure[i]) # 2 = hidden neuron
                    i = i + 1
                        
                # add in the outputs as the last layer
                neurons.append(Outs)

                # Extract the weights for the input neurons to pain layer
                line = f.readline().strip('\n')
                weights = self._parseWeights(line = line)

                # connect all the input neurons to the pain neurons
                self.fillConnects(fromLayer=neurons[0], toLayer=neurons[1], weights=weights)
                
                if numHideLays > 0:
                    # extract the weights for the inputs to first hidden layer
                    line = f.readline().strip('\n')
                    weights = self._parseWeights(line = line)

                    # connect all the input neurons to the first hidden layer, if it exists
                    self.fillConnects(fromLayer=neurons[0], toLayer=neurons[2], weights = weights)

                    # connect pain neurons to all hidden layers
                    hlayer = 2 # 0th hidden layer
                    while hlayer - 1 < numHideLays: # account for 2 offset
                        # Extract weights for pain neurons to each hidden layer
                        line = f.readline().strip('\n')
                        weights = self._parseWeights(line = line)

                        self.fillConnects(fromLayer=neurons[1], toLayer=neurons[hlayer], weights=weights)
                        hlayer = hlayer + 1

                    # fill in the rest of the layers as normal
                    layer = 2 # 0th hidden layer
                    while layer - 1 < numHideLays:
                        # Extract weights for current hidden layer to next layer
                        line = f.readline().strip('\n')
                        weights = self._parseWeights(line = line)

                        self.fillConnects(fromLayer=neurons[layer], toLayer=neurons[layer + 1], weights = weights)
                        layer = layer + 1

                self.neurons = neurons
                
        except Exception as e:
            raise e

    def _parseWeights(self, line : str) -> list:
        """
            Parses the weights out of a string
            For loading in the network from a file
        """
        breakByNeuron = line.split(';')
        # at this point, breakByNeuron has a list of strings, 
        # each of which contains the weights for each synapse from that neuron, separated by spaces
        # function to split each string around the spaces
        splitWeights = lambda x : x.split(' ')
        layerWeights = list(map(splitWeights, breakByNeuron))
        
        # list to start adding the weights to
        weightsByNeuron = list()
        for i in range(0, len(layerWeights)):
            neuronWeights = layerWeights[i]
            # extract the sublists of synapse weights for this neuron
            weights = list()
            for weight in neuronWeights:
                try:
                    weights.append(float(weight))
                except ValueError as e:
                    # if it tries to convert a space, leave it
                    pass
            # Add the extracted weights to the weights by neuron
            weightsByNeuron.append(weights)

        return weightsByNeuron
       

    def writeNetwork(self, path : str):
        """
            Write the network to a file 
            file formatted as (with spaces):
                phase_duration dt maxI
                ispike0 ispike1 ispike2 ...
                #inputs #pains #outputs #hidden0 #hidden1 ...
                in0>pain0 in0>pain1 in0>pain2 ; in1>pain0 in1>pain1 in1>pain2 ; ...
                in0>hidden0_0 in0>hidden0_1 in0>hidden0_2 ; in1>hidden0_0 in1>hidden0_1 in1>hidden0_2 ; ...
                pain0>hidden0_0 pain0>hidden0_1 pain0>hidden0_2 ; pain1>hidden0_0 pain1>hidden0_1 pain1>hidden0_2 ; ...
                pain0>hidden1_0 pain0>hidden1_1 pain0>hidden1_2 ; pain1>hidden1_0 pain1>hidden1_1 pain1>hidden1_2 ; ...
                ...
                hidden0_0>hidden1_0 hidden0_0>hidden1_1 hidden0_0>hidden1_2 ; hidden0_1>hidden1_0 hidden0_1>hidden1_1 ...
                ...
                hiddenN_0>out0 hiddenN_0>out1 hiddenN_0>out2 ; hiddenN_1>out0 hiddenN_1>out1 hiddenN_1>out2 ; ...

        """

        with open(path, "w") as f:
            # Write the network parameters
            f.write(f"{self.phaseDuration} {self.dt} {self.maxI}\n")
            # Write the current spike shape
            f.write(_list2str(self.ispikeshape)+"\n")

            # Write the structure shape
            f.write(_list2str(self.structure)+"\n")
            
            # now go through the input layer
            line1 = "" # to pain layer
            line2 = "" # to hidden layer 0
            for neu in self.neurons[0]:
                neuOutSyns = neu.outSyns
                for syn in neuOutSyns:
                    # if it's going to a pain, add to line 1
                    if syn.post.type == _PAIN:
                        line1 = line1 + str(syn.weight) + " "
                    elif syn.post.type == _HIDDEN:
                        line2 = line2 + str(syn.weight) + " "

                # check if this is the last neuron
                if neu is not self.neurons[0][-1]:
                    # if not, add a semicolon to the strings
                    line1 = line1 + ";"
                    line2 = line2 + ";"
            
            # Now that we've constructed the inputs to pain and inputs to hidden layer 0 lines, write to file
            f.write(line1 + "\n")
            f.write(line2 + "\n")

            # Now iterate through the pain neurons
            # Some math is going to need to be done here
            startInd = 0
            for hlSize in self.structure[3:]:
                # this is going to start pulling out the size of hidden layers
                # extract that many output syns from each neuron
                # start the next time with extracting from startInd
                line = ""
                for neu in self.neurons[1]:
                    neuOutSyns = neu.outSyns
                    for i in range(0, hlSize - 1):
                        # pulls out hlSize synapses from each pain neuron
                        line = line + neuOutSyns[startInd + i] + " "

                    # check if this is the last pain neuron
                    if neu is not self.neurons[1][-1]:
                        # if not, add a semicolon to the strings
                        line = line + ";"
                        
                # increase the start Index for the next time through
                startInd = startInd + hlSize
                # print line to file
                f.write(line + "\n")
            
            # Alright, pain neurons are done.  Now do the hidden layers as normal
            # find the number of hidden layers; minus 3 for input, output, and pain layers
            numHideLays = len(self.structure) - 3
            layer = 2 # 0th hidden layer
            while layer - 1 < numHideLays:
                # Extract and write weights for current hidden layer to next layer
                line = ""
                for neu in self.neurons[layer]:
                    for syn in neu.outSyns:
                        line = line + str(syn.weight) + " "
                    if neu is not self.neurons[layer][-1]:
                        # not the final, add a semicolon to the strings
                        line = line + ";"
                f.write(line + "\n")
                layer = layer + 1

            f.close()
                    


    

    def buildNetwork(self, structure : list):
        """
            Builds the network based on the structure
            Inputs:
                structure   - list containing number of neurons in each layer of the network.
                              [inputs, pain neurons, outputs, hidden layer 1, hidden layer 2,...]
            Outputs:
                neurons     - list of Neurons, according to shape of structure
        """
        # Extract dimensions for the network; to ease readability
        numIns      = structure[0]
        numPains    = structure[1]
        numOuts     = structure[2]
        
        # find the number of hidden layers; minus 3 for input, output, and pain layers
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
        #ispikeTotal = funcs.ispike(dt=self.dt)
        #ispikeshape = ispikeTotal['current']

        # connect all the input neurons to the pain neurons
        self.fillConnects(fromLayer=neurons[0], toLayer=neurons[1])
        
        if numHideLays > 0:
            # connect all the input and pain neurons to the first hidden layer, if it exists
            self.fillConnects(fromLayer=neurons[0], toLayer=neurons[2])

            # connect pain neurons to all hidden layers
            hlayer = 2 # 0th hidden layer
            while hlayer - 1 < numHideLays: # account for 2 offset
                self.fillConnects(fromLayer=neurons[1], toLayer=neurons[hlayer])
                hlayer = hlayer + 1

            # fill in the rest of the layers as normal
            layer = 2 # 0th hidden layer
            while layer - 1 < numHideLays:
                self.fillConnects(fromLayer=neurons[layer], toLayer=neurons[layer + 1])
                layer = layer + 1

        return neurons
    
    def fillConnects(self, fromLayer : list, toLayer : list, weights : list = None):
        """
            initializes all connections from neurons in fromLayer to neurons in toLayer

            Inputs:
                fromLayer   - list of neurons in presynaptic layer
                toLayer     - list of neurons in postsynaptic layer
                weights     - list of weights, if loading in from a file
            
            Outputs:
                None
        """
        if weights is None:
            rng.seed() # seed with current time, or other random seed from the OS
        # rng.random() * self.maxI to get a value
        
        #if fromLayer[0].type != _PAIN:
        # Connect the layers
        i = 0
        for fromNeu in fromLayer:
            j = 0 # synapse weights tracker
            for toNeu in toLayer:
                # fromNeu is the presynaptic connection of the toNeu
                if weights is None:
                    # No weights provided: randomize the weights
                    fromNeu.connect(toNeu, 0, ispike=self.ispike, weight=rng.random() * self.maxI)
                else:
                    # weights provided
                    fromNeu.connect(toNeu, 0, ispike=self.ispike, weight = weights[i][j])
                    
                    # track synapse weights for this fromNeuron
                    j = j + 1
            # get synapse weights for the next neuron in the from layer
            i = i + 1
        """
        else:
            # This is a pain neuron- some math needs to be done here
            #TODO
            # Connect the layers
            i = 0
            for painNeu in fromLayer:
                j = 0 # synapse weights tracker
                for toNeu in toLayer:
                    # painNeu is the presynaptic connection of the toNeu
                    if weights is None:
                        # No weights provided: randomize the weights
                        fromNeu.connect(toNeu, 0, ispike=self.ispike, weight=rng.random() * self.maxI)
                    else:
                        # weights provided
                        
                        fromNeu.connect(toNeu, 0, ispike=self.ispike, weight = weights[i][j])
                        
                        # track synapse weights for this fromNeuron
                        j = j + 1
                # get synapse weights for the next neuron in the from layer
                i = i + 1
            """
                

            

        
    
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
    
