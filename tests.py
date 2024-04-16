from synapse import Synapse
from neuron import Neuron
import funcs
import numpy as np
import matplotlib.pyplot as plt

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

    # Make sure neurons both register the synapse connection properly
    if inNeuron.outSyns.pop() is not outNeuron.inSyns.pop():
        print(f"FAILED: Synapse Object mismatch")
    else:
        print(f"PASSED: Synapses match")

    # Ensure neurons can see each other
    
    if inNeuron.outSyns.

    return

""" SYNAPSE TESTS """
def synInit():
    test_pre_neu = Neuron(type=_INPUT)
    test_syn = Synapse()

if __name__ == "__main__":
    # Test to Run
    neuInit()
    pass


    

