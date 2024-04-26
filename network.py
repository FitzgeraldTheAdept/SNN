import numpy as np
import random as rng
#import matplotlib.pyplot as plt
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
            outStr = outStr + str(item) + " "
        
        
        return outStr

def _line2list(line : str, toType : type = float) -> list:
    """
        Converts a given string, read from file, into a float
        The string 
        Handles empty strings
        INPUTS:
            line - input line to convert to a list of floats
    """
    # first, split the line based on spaces
    splitVals = line.split(' ')
    outList = list()
    
    for val in splitVals:
        try:
            outList.append(toType(val))
        except ValueError as e:
            # if fails, ignore it
            pass

    return outList

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
        weightTime      - time to let network settle before finding activities
    """
    def __init__(self, 
                 path : str = None,
                 phaseDuration : int = 100, 
                 dt : float = 0.1, 
                 structure : list = [2, 1, 1], 
                 simStep : int = 0,
                 maxI : float = 40,
                 waitTime = 20):
        
        if path is None:
            # build a new network
            self.phaseDuration = phaseDuration
            self.waitTime = waitTime
            self.dt            = dt
            self.structure     = structure

            # create the current spike shape
            ispikeTotal = funcs.ispike(dt=self.dt)
            self.ispikeshape = ispikeTotal['current']
            self.maxI = maxI

            self.t             = list(map(lambda x: x * self.dt, range(0, int(self.phaseDuration / self.dt) + 1,1)))
            self.simStep       = simStep
            self.neurons       = self.buildNetwork(self.structure)

            
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
        self.simStep = 0
        try:
            with open(path, 'r') as f:
                # first line is formatted as:
                # phase_duration dt maxI
                line = f.readline().strip('\n')
                netParams = line.split(' ')
                                
                self.phaseDuration = int(netParams[0])
                self.dt            = float(netParams[1])
                self.maxI          = float(netParams[2])
                self.waitTime      = int(netParams[3])

                self.t             = list(map(lambda x: x * self.dt, range(0, int(self.phaseDuration / self.dt) + 1,1)))

                # Read the current spike shape from the second line
                line = f.readline().strip('\n')
                self.ispikeshape = _line2list(line=line, toType = float)

                # Read in the structure from the third line
                line = f.readline().strip('\n')
                
                self.structure = _line2list(line=line, toType = int)
                
                
                # Extract dimensions for the network; to ease readability
                numIns      = self.structure[0]
                numPains    = self.structure[1]
                numOuts     = self.structure[2]
                
                # find the number of hidden layers; minus 3 for input, output, and pain layers
                numHideLays = len(self.structure) - 3

                Ins = list()
                Pains = list()
                Outs = list()

               # Build network of distinct neurons

                for i in range(0, numIns):
                    Ins.append(Neuron(type=1))      # type 1 = input
                for i in range(0, numPains):
                    Pains.append(Neuron(type=-1))   # type -1 = pain
                for i in range(0, numOuts):
                    Outs.append(Neuron(type=0))     # type 0 = output

                neurons = [Ins, Pains]
                
                # Add in hidden layers
                i = 3
                while i < len(self.structure):
                    HL = list()
                    for j in range(0, self.structure[i]):
                        HL.append(Neuron(type=2))  # 2 = hidden neuron
                    neurons.append(HL)
                    i = i + 1
                        
                # add in the outputs as the last layer
                neurons.append(Outs)
                
                # Extract the weights for the input neurons to pain layer
                #line = f.readline().strip('\n')
                #weights = self._parseWeights(line = line)

                # connect all the input neurons to the pain neurons
                # self.fillConnects(fromLayer=neurons[0], toLayer=neurons[1], weights=weights)
                
                if numHideLays > 0:
                    # extract the weights for the inputs to first hidden layer
                    line = f.readline().strip('\n')
                    weights = self._parseWeights(line = line)

                    # connect all the input neurons to the first hidden layer, if it exists
                    self.fillConnects(fromLayer=neurons[0], toLayer=neurons[2], weights = weights)

                    # connect pain neurons to all hidden layers
                    hlayer = 2 # 0th hidden layer
                    while hlayer - 1 <= numHideLays: # account for 2 offset
                        # Extract weights for pain neurons to each hidden layer
                        line = f.readline().strip('\n')
                        weights = self._parseWeights(line = line)
                        

                        self.fillConnects(fromLayer=neurons[1], toLayer=neurons[hlayer], weights=weights)
                        hlayer = hlayer + 1

                    # fill in the rest of the layers as normal
                    layer = 2 # 0th hidden layer
                    while layer - 1 <= numHideLays:
                        # Extract weights for current hidden layer to next layer
                        line = f.readline().strip('\n')
                        weights = self._parseWeights(line = line)
                        

                        self.fillConnects(fromLayer=neurons[layer], toLayer=neurons[layer + 1], weights = weights)
                        layer = layer + 1

                else:
                    # connect inputs and pain to outputs directly
                    # extract the weights for the inputs to outputs
                    line = f.readline().strip('\n')
                    weights = self._parseWeights(line = line)

                    # connect all the input neurons to outputs
                    self.fillConnects(fromLayer=neurons[0], toLayer=neurons[2], weights = weights)

                    # connect pain neurons to outputs
                    line = f.readline().strip('\n')
                    weights = self._parseWeights(line = line)

                    #self.fillConnects(fromLayer=neurons[1], toLayer=neurons[2], weights = weights)

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
        for i in range(0, len(layerWeights) - 1):
            neuronWeights = layerWeights[i]
           
            # extract the sublists of synapse weights for this neuron
            weights = list()
            for weight in neuronWeights:
                if weight != '':
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
                in0>pain0 in0>pain1 in0>pain2 ;in1>pain0 in1>pain1 in1>pain2 ; ... ;
                in0>hidden0_0 in0>hidden0_1 in0>hidden0_2 ;in1>hidden0_0 in1>hidden0_1 in1>hidden0_2 ; ... ;
                pain0>hidden0_0 pain0>hidden0_1 pain0>hidden0_2 ;pain1>hidden0_0 pain1>hidden0_1 pain1>hidden0_2 ; ... ;
                pain0>hidden1_0 pain0>hidden1_1 pain0>hidden1_2 ;pain1>hidden1_0 pain1>hidden1_1 pain1>hidden1_2 ; ... ;
                ...
                hidden0_0>hidden1_0 hidden0_0>hidden1_1 hidden0_0>hidden1_2 ;hidden0_1>hidden1_0 hidden0_1>hidden1_1 ... ;
                ...
                hiddenN_0>out0 hiddenN_0>out1 hiddenN_0>out2 ;hiddenN_1>out0 hiddenN_1>out1 hiddenN_1>out2 ; ... ;

        """

        # DEBUG
        self.writeNetwork2(path=path)
        return

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
                        # We've got a pain output
                        
                        line1 = line1 + str(syn.weight) + " "
                    elif syn.post.type == _HIDDEN:
                        
                        line2 = line2 + str(syn.weight) + " "

                # check if this is the last neuron
                # add a semicolon to the strings
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
                    
                    for i in range(0, hlSize):
                        # pulls out hlSize synapses from each pain neuron
                        line = line + str(neuOutSyns[startInd + i].weight) + " "

                    # check if this is the last pain neuron
                    # add a semicolon to the strings
                    line = line + ";"
                        
                # increase the start Index for the next time through
                startInd = startInd + hlSize
               
                
                f.write(line + "\n")
            
            # Alright, pain neurons are done.  Now do the hidden layers as normal
            # find the number of hidden layers; minus 3 for input, output, and pain layers
            numHideLays = len(self.structure) - 3
            layer = 2 # 0th hidden layer
            while layer - 1 <= numHideLays:
                
                # Extract and write weights for current hidden layer to next layer
                line = ""
                for neu in self.neurons[layer]:
                    
                    for syn in neu.outSyns:
                        line = line + str(syn.weight) + " "
                    # add a semicolon to the strings
                    line = line + ";"
                
                f.write(line + "\n")
                layer = layer + 1

            f.close()

    def writeNetwork2(self, path : str):
        """
            Write the network to a file.  Severed input to pain connections
            file formatted as (with spaces):
                phase_duration dt maxI
                ispike0 ispike1 ispike2 ...
                #inputs #pains #outputs #hidden0 #hidden1 ...
                in0>hidden0_0 in0>hidden0_1 in0>hidden0_2 ;in1>hidden0_0 in1>hidden0_1 in1>hidden0_2 ; ... ;
                pain0>hidden0_0 pain0>hidden0_1 pain0>hidden0_2 ;pain1>hidden0_0 pain1>hidden0_1 pain1>hidden0_2 ; ... ;
                pain0>hidden1_0 pain0>hidden1_1 pain0>hidden1_2 ;pain1>hidden1_0 pain1>hidden1_1 pain1>hidden1_2 ; ... ;
                ...
                hidden0_0>hidden1_0 hidden0_0>hidden1_1 hidden0_0>hidden1_2 ;hidden0_1>hidden1_0 hidden0_1>hidden1_1 ... ;
                ...
                hiddenN_0>out0 hiddenN_0>out1 hiddenN_0>out2 ;hiddenN_1>out0 hiddenN_1>out1 hiddenN_1>out2 ; ... ;

        """

        with open(path, "w") as f:
            # Write the network parameters
            f.write(f"{self.phaseDuration} {self.dt} {self.maxI} {self.waitTime}\n")
            # Write the current spike shape
            f.write(_list2str(self.ispikeshape)+"\n")
            
            # Write the structure shape
            f.write(_list2str(self.structure)+"\n")
            
            
            # now go through the input layer
            line= "" # to hidden layer 0
            
            for neu in self.neurons[0]:
                neuOutSyns = neu.outSyns
               
                
                for syn in neuOutSyns:            
                    line = line + str(syn.weight) + " "

                # check if this is the last neuron
                # add a semicolon to the strings
                line = line + ";"
            
            # Now that we've constructed the inputs to pain and inputs to hidden layer 0 lines, write to file
            f.write(line + "\n")

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
                    
                    for i in range(0, hlSize):
                        # pulls out hlSize synapses from each pain neuron
                        line = line + str(neuOutSyns[startInd + i].weight) + " "

                    # check if this is the last pain neuron
                    # add a semicolon to the strings
                    line = line + ";"
                        
                # increase the start Index for the next time through
                startInd = startInd + hlSize
               
                
                f.write(line + "\n")
            
            # Alright, pain neurons are done.  Now do the hidden layers as normal
            # find the number of hidden layers; minus 3 for input, output, and pain layers
            numHideLays = len(self.structure) - 3
            layer = 2 # 0th hidden layer
            while layer - 1 <= numHideLays:
                
                # Extract and write weights for current hidden layer to next layer
                line = ""
                for neu in self.neurons[layer]:
                    
                    for syn in neu.outSyns:
                        line = line + str(syn.weight) + " "
                    # add a semicolon to the strings
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

        Ins = list()
        Pains = list()
        Outs = list()

        # Build network of distinct neurons

        for i in range(0, numIns):
            Ins.append(Neuron(type=1))      # type 1 = input
        for i in range(0, numPains):
            Pains.append(Neuron(type=-1))   # type -1 = pain
        for i in range(0, numOuts):
            Outs.append(Neuron(type=0))     # type 0 = output
    
        neurons = [Ins, Pains]
        
        # Add in hidden layers
        i = 3
        while i < len(structure):
            HL = list()
            for j in range(0, structure[i]):
                HL.append(Neuron(type=2))   # type 2 = hidden neuron

            neurons.append(HL) 
            i = i + 1
                
        # add in the outputs as the last layer
        neurons.append(Outs)

        # define the spike shape
        #ispikeTotal = funcs.ispike(dt=self.dt)
        #ispikeshape = ispikeTotal['current']

        rng.seed() # seed with current time, or other random seed from the OS
        
        # connect all the input neurons to the pain neurons
        # DEBUG: SEVER INPUT TO PAIN CONNECTIONS
        #self.fillConnects(fromLayer=neurons[0], toLayer=neurons[1])
            
        
        if numHideLays > 0:
            # connect all the input and pain neurons to the first hidden layer, if it exists
            self.fillConnects(fromLayer=neurons[0], toLayer=neurons[2])

            # connect pain neurons to all hidden layers
            hlayer = 2 # 0th hidden layer
            while hlayer - 1 <= numHideLays: # account for 2 offset
                self.fillConnects(fromLayer=neurons[1], toLayer=neurons[hlayer])
                hlayer = hlayer + 1

            # fill in the rest of the layers as normal
            layer = 2 # 0th hidden layer
            while layer - 1 <= numHideLays:
                self.fillConnects(fromLayer=neurons[layer], toLayer=neurons[layer + 1])

                layer = layer + 1
        else:
            # no hidden layers
            self.fillConnects(fromLayer=neurons[0], toLayer=neurons[2])
            self.fillConnects(fromLayer=neurons[1], toLayer=neurons[2])

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
                    fromNeu.connect(toNeu, 0, ispike=self.ispikeshape, weight=rng.random() * self.maxI)
                else:
                    # weights provided
                    
                    fromNeu.connect(toNeu, 0, ispike=self.ispikeshape, weight = weights[i][j])
                    
                    # track synapse weights for this fromNeuron
                    j = j + 1
            # get synapse weights for the next neuron in the from layer
            i = i + 1


    def dumpInfo(self):
        """
            DEBUG Function: dumps literally all the info
                   
        """
        print(f"structure = {self.structure}")
        i = 0
        for layer in self.neurons:
            print(f"layer {i} has {len(layer)} neurons")
            i = i + 1
            for neu in layer:
                print(f"{neu.type} neuron has output synapses with the following weights:")
                for syn in neu.outSyns:
                    print(syn.weight)
    
    def drawNetwork(self):
        """
            Draws the Network diagram
        """
        for i in self.neurons:
            print(self.structure)

    def phase(self, I_in : list, I_pain : list = None, I_out : list = None):
        """
            Execute a phase (or a propagation).  
            In other words, solve the whole network for every simstep
            Inputs:
                I_in - currents derived from the input strength, list of floats
                I_pain - currents to the pain neurons
                I_out - currents to the output neurons
        """
        # reset all the neurons
        for layer in self.neurons:
            for neu in layer:
                neu.reset()
        
        self.simStep = 0 # reset
        scaled_Iin = list(np.asarray(I_in) * self.maxI)
        print(f"NETWORK: Scaled input currents are: {scaled_Iin}")
        if I_pain is not None:
            scaled_Ipain = list(np.asarray(I_pain) * self.maxI)
            #print(f"NETWORK: Scaled pain currents are: {scaled_Ipain}")

        if I_out is not None:
            scaled_Iout = list(np.asarray(I_out) * self.maxI)
            
            print(f"NETWORK: Scaled output currents are: {scaled_Iout}")

        while self.simStep < len(self.t):

            # Solve all the neurons, starting with the input layer and moving forward
            for layer in self.neurons:
                # input/pain current index tracker
                i = 0
                for neu in layer:
                    if neu.type == _INPUT:
                                                
                        neu.step(simStep = self.simStep, dt = self.dt, I_in = scaled_Iin[i])
                        
                    elif neu.type == _PAIN:
                        # pain neuron
                        if I_pain is not None:
                            neu.step(simStep = self.simStep, dt = self.dt, I_in = scaled_Ipain[i])
                        else:
                            neu.step(simStep = self.simStep, dt = self.dt, I_in = None)
                        # Neurons will check synapses to find their other input current
                    elif neu.type == _OUTPUT and I_out is not None:
                        # training- output currents applied
                        neu.step(simStep=self.simStep, dt=self.dt, I_in = scaled_Iout[i])
                    else:
                        # hidden layer neuron or output neuron out of training
                        neu.step(simStep = self.simStep, dt = self.dt, I_in = None)
                        # Neurons will check synapses to find their input current
                    i = i + 1
                    
            
            # increment to next simulation step
            self.simStep = self.simStep + 1
        print(f"NETWORK: outputs are: {self.getOuts()}")
        

    def adjustWeights(self, lr : float):
        """
            Adjusts the weights of the network using hebbian learning rules in trainer file
            INPUTS:
                lr - learning rate (on the order of 1)
        
        """
        # Go through each neuron in each layer, finding the hebbian coefficient of each synapse.  
        # Multiply that by the learning rate, adjust the weight by that much
        
        for lay in self.neurons:
            
            for neu in lay:
                if neu.type != _PAIN and neu.type != _OUTPUT:
                    neu.adjustWeights(lr, dt=self.dt, pD=self.phaseDuration, iDur=self.waitTime)
            
            #for neu in lay:
            #   neu.adjustWeights(lr, dt=self.dt, pD=self.phaseDuration, iDur=self.waitTime) 
        
        

    def getOuts(self) -> list:
        """
            Calculates a vector of the same length as the number of outputs, 
            containing the activity of each output neuron

            OUTPUTS:
                a list of activity values between 0 and 1
        """
        if self.simStep > 0 : # DEBUG : change from len(self.t)
            # simulation has been run
            outs = list()
            for x in self.neurons[-1]:
                outs.append(x.getAct(dt=self.dt, pD = self.phaseDuration, iDur = self.waitTime))
            
            # DEBUG
            
            return outs
        else:
            return None


    def plotOuts(self):
        # For debugging
        import matplotlib.pyplot as plt
        plt.figure()

        neusPlots = list()
        #iPlots = list()
        for neu in self.neurons[-1]:
            neusPlots.append(neu.v)
        
        #for neu in self.neurons[2]:
        #    iPlots.append(neu.I)

        for i in range(0, len(neusPlots)):
            plt.plot(self.t, neusPlots[i][0:-1])   
            #plt.plot(self.t, iPlots[i][0:-1])

        names = list()
        for i in range(0, len(self.neurons[-1])):
            names.append(f"Out {i}")
            #names.append(f"In {i} I")
        
        plt.legend(names)
        #plt.legend()
        plt.show()
        

#if __name__ == '__main__':
#   net = Network(structure=[1,2,3,4,5,6])
#   net.drawNetwork()
    
