from synapse import Synapse
from neuron import Neuron
import funcs
import numpy as np
import matplotlib.pyplot as plt
import random
import math

_INPUT = 1
_OUTPUT = 0
_HIDDEN = 2
_PAIN = -1

""" TEST TO RUN """
test2Run = "synInit"

""" NEURON TESTS """
def neuInit():
    """
        Make sure the neurons actually initializes correctly.
        - Initialize two neurons, one input, one output
        - Connect the Neurons with a synapse
        - Ensure Neuron A can find Neuron B via Synapse
    """
    # Create both neurons
    inNeuron = Neuron(type=_INPUT)
    outNeuron = Neuron(type=_OUTPUT)

    # Connect the neurons
    inNeuron.connect(outNeuron, 0)

    # make a list of the synapses
    a_syns = list(inNeuron.outSyns)
    b_syns = list(outNeuron.inSyns)

    # Make sure neurons both register the synapse connection properly
    if a_syns[0] is not b_syns[0]:
        print(f"FAILED: Synapse Object mismatch")
    else:
        print(f"PASSED: Synapses match")

    # Ensure neurons can see each other
    
    if a_syns[0].post is outNeuron:
        print(f"PASSED: In Neuron Can reach Out Neuron")
    else:
        print(f"FAILED: In Neuron Cannot reach Out Neuron")

    return

def synAdj():
    """
        Ensure that the synapse weight adjustment works
    """
    pass

def neuCalcI():
    """
        test _calcI function of the neuron
        - Connect two neurons to the output neuron
        - randomize spike timings of the input neurons
        - display the input current 

    """
    # initialize neurons
    in_a = Neuron(type=_INPUT)
    in_b = Neuron(type=_INPUT)
    out_a = Neuron(type=_OUTPUT)

    # Set the random number generator seed for the initial synapse weights
    random.seed(42)

    # generate spike shape
    d_t = 0.1
    i_spike_total = funcs.ispike(dt = d_t)
    i_spike_shape = i_spike_total['current']

    # Connect neurons, generate synapse weights
    in_a.connect(toNeuron=out_a, prePost=0, ispike=i_spike_shape)
    in_b.connect(toNeuron=out_a, prePost=0, ispike=i_spike_shape)

    # set the rng seed for spike timings
    random.seed(21)
    t = funcs.floatRange(0, 100, d_t)
    T = len(t)
    spike_chance = 0.01 # chance of spike, 0.3 = 30%

    s1 = np.empty(T)
    s2 = np.empty(T)
    spikes1 = list()
    spikes2 = list()
    for i in range(0, T - 1):
        # Define random spike timings
        s1[i] = math.floor(random.random() + spike_chance)
        s2[i] = math.floor(random.random() + spike_chance)
        
        # translate to spike vectors
        if s1[i] == 1:
            spikes1.append(i)
        if s2[i] == 1:
            spikes2.append(i)

    # plot the spikes
    plt.figure()
    plt.plot(range(0, T), s1, 'r.' )
    plt.xlabel("Time Step")
    plt.ylabel("Spike")
    plt.title("Neuron 1 Spike timings")
    plt.grid()
    

    in_a.spikes = spikes1
    in_b.spikes = spikes2

    # Check the current through the synapse from in_a to out_a
    in_a_syns = list(in_a.outSyns)
    syn = in_a_syns[0]
    syn_I = np.empty(T)
    for i in range(0, T):
        syn_I[i] = syn.step(i)

        
    # plot the current 
    plt.figure()
    plt.plot(range(0, T), syn_I, 'r.' )
    plt.xlabel("Time Step")
    plt.ylabel("Current (I)")
    plt.title("in_a to out_b synapse current")
    plt.grid()
    plt.show()
    pass

""" SYNAPSE TESTS """
def synInit():
    test_pre_neu = Neuron(type=_INPUT)
    test_syn = Synapse()

if __name__ == "__main__":
    # Test to Run
    neuCalcI()
    pass


    

