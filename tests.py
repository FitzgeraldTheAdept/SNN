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
    random.seed(41)

    # generate spike shape
    d_t = 0.01
    i_spike_total = funcs.ispike(dt = d_t, holdTime = 2)
    i_spike_shape = i_spike_total['current']

    # Connect neurons, generate synapse weights
    in_a.connect(toNeuron=out_a, prePost=0, ispike=i_spike_shape)
    in_b.connect(toNeuron=out_a, prePost=0, ispike=i_spike_shape)

    # set the rng seed for spike timings
    random.seed(21)
    t = funcs.floatRange(0, 100, d_t)
    T = len(t)
    spike_chance = 0.001 # chance of spike, 0.3 = 30%

    s1 = np.empty(T)
    s2 = np.empty(T)
    spikes1 = list()
    spikes2 = list()
    for i in range(0, T):
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
    in_b_syns = list(in_b.outSyns)
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

    neu_I = np.empty(T)
    # neu_v = np.empty(T)
    # Run the neuron calcI across the time steps
    for i in range(0, T):
        neu_I[i] = out_a._calcI(simStep=i, dt=d_t)
        out_a.step(simStep=i, dt=d_t )

    print(f"{in_b_syns[0].weight}")      
    print(f"{in_a_syns[0].weight}")     

    # plot the current 
    plt.figure()
    plt.plot(range(0, T), neu_I, 'r.' )
    plt.xlabel("Time Step")
    plt.ylabel("Current (I)")
    plt.title("out_a neuron input current")
    plt.grid()
   
    
    plt.figure()
    plt.plot(range(0,T), out_a.v[1:len(out_a.v)])
    plt.xlabel("Time Step")
    plt.ylabel("Voltage (V)")
    plt.title("out_a neuron output Voltage")
    plt.grid()
    plt.show()


def inTest():
    """
        Test to see if input neurons work
    """
    # initialize neurons
    in_a = Neuron(type=_INPUT)
    in_b = Neuron(type=_INPUT)
    out_a = Neuron(type=_OUTPUT)

    # Set the random number generator seed for the initial synapse weights
    random.seed(41)

    # generate spike shape
    d_t = 0.01
    i_spike_total = funcs.ispike(dt = d_t)
    i_spike_shape = i_spike_total['current']

    # Connect neurons, generate synapse weights
    in_a.connect(toNeuron=out_a, prePost=0, weight=20.0, ispike=i_spike_shape)
    in_b.connect(toNeuron=out_a, prePost=0, weight=20.0, ispike=i_spike_shape)

    # Make time series data
    t = funcs.floatRange(0, 100, d_t)
    T = len(t)

    # start stepping the network
    for simStep in range(0, T):
        # Step the input neurons with input currents defined 
        in_a.step(simStep=simStep, dt = d_t, I_in = 20)
        in_b.step(simStep=simStep, dt = d_t, I_in = 30)
        
        # Step the output neurons
        out_a.step(simStep=simStep, dt = d_t)
    
    # Plot the time series data with membrane voltage
    plt.figure()
    plt.plot(t, out_a.v[1:len(out_a.v)], 'r-')
    plt.plot(t, in_a.v[1:len(in_a.v)], 'b-')
    plt.plot(t, in_b.v[1:len(in_b.v)], 'g-')
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Voltage (V)")
    plt.title("Neuron Voltages")
    plt.legend(["Output", "Input A", "Input B"])
    plt.grid()
    plt.show()

""" PAIN TESTS """
def painI():
    """
        Test for Pain neurons
        - Check model output with two input connections, one pain and one input
    """
    # initialize neurons
    in_a = Neuron(type=_INPUT)
    in_b = Neuron(type=_PAIN)
    out_a = Neuron(type=_OUTPUT)

    # Set the random number generator seed for the initial synapse weights
    random.seed(41)

    # generate spike shape
    d_t = 0.01
    i_spike_total = funcs.ispike(dt = d_t, rt = 2, ft = 50, holdTime=2)
    i_spike_shape = i_spike_total['current']

    # Connect neurons, generate synapse weights
    in_a.connect(toNeuron=out_a, prePost=0, weight=20.0, ispike=i_spike_shape)
    in_b.connect(toNeuron=out_a, prePost=0, weight=20.0, ispike=i_spike_shape)

    # Make time series data
    t = funcs.floatRange(0, 100, d_t)
    T = len(t)

    # start stepping the network
    for simStep in range(0, T):
        # Step the input neurons with input currents defined 
        in_a.step(simStep=simStep, dt = d_t, I_in = 30)
        in_b.step(simStep=simStep, dt = d_t, I_in = 20)
        
        # Step the output neurons
        out_a.step(simStep=simStep, dt = d_t)
    
    # Plot the time series data with membrane voltage
    plt.figure()
    plt.plot(t, out_a.v[1:len(out_a.v)], 'r-')
    plt.plot(t, in_a.v[1:len(in_a.v)], 'b-')
    plt.plot(t, in_b.v[1:len(in_b.v)], 'g-')
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Voltage (V)")
    plt.title("Neuron Voltages")
    plt.legend(["Output", "Normal Input", "Pain Input"])
    plt.grid()
    plt.show()

def painIntoPain():
    """
        Test for Pain neurons
        - Input coupled to pain, both coupled to output
    """
    # initialize neurons
    in_a = Neuron(type=_INPUT)
    pain_b = Neuron(type=_PAIN)
    out_a = Neuron(type=_OUTPUT)

    # Set the random number generator seed for the initial synapse weights
    random.seed(41)

    # generate spike shape
    d_t = 0.01
    i_spike_total = funcs.ispike(dt = d_t, rt = 2, ft = 50, holdTime=2)
    i_spike_shape = i_spike_total['current']

    # Connect input and pain to the output
    in_a.connect(toNeuron=out_a, prePost=0, weight=20.0, ispike=i_spike_shape)
    pain_b.connect(toNeuron=out_a, prePost=0, weight=20.0, ispike=i_spike_shape)
    # Connect the input to the pain neuron
    in_a.connect(toNeuron=pain_b, prePost = 0, weight=20.0, ispike=i_spike_shape)

    # Make time series data
    t = funcs.floatRange(0, 100, d_t)
    T = len(t)

    # start stepping the network
    for simStep in range(0, T):
        # Step the input neurons with input currents defined 
        in_a.step(simStep=simStep, dt = d_t, I_in = 30)
        pain_b.step(simStep=simStep, dt = d_t, I_in = 20)
        
        # Step the output neurons
        out_a.step(simStep=simStep, dt = d_t)
    
    # Plot the time series data with membrane voltage
    plt.figure()
    plt.plot(t, out_a.v[1:len(out_a.v)], 'r-')
    plt.plot(t, in_a.v[1:len(in_a.v)], 'b-')
    plt.plot(t, pain_b.v[1:len(pain_b.v)], 'g-')
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Voltage (V)")
    plt.title("Neuron Voltages")
    plt.legend(["Output", "Normal Input", "Pain Input"])
    plt.grid()
    plt.show()

""" SYNAPSE TESTS """
def synInit():
    test_pre_neu = Neuron(type=_INPUT)
    test_syn = Synapse()

if __name__ == "__main__":
    # Test to Run
    #neuCalcI()
    #inTest()
    #painI()
    painIntoPain()
    pass


    

