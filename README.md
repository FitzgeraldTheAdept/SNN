# SNN
Spiking Neural Network 
Benjamin Wittrup


## Instructions

Run 
``` 
controller.py
```

## TODO:
Switch Neuron Model to correlate input current to membrane potential, determine if the neuron is active (threshold voltage) and how active it is based off of input current.

## RECOMMENDATIONS

Needed nonlinear synapse weight adjustment
- to help prevent all synapses from firing and saturating the input current too quickly
- also a shorter current spike may have helped
- also using a smaller maximum synapse current (~10)
- rather than amplifying pain current directly, utilize multiple pain neurons with similar rules
- Evaluate output currents, not current to the outputs
- Train with a null case
- Make the feedback neurons make already active neurons more active, and less active neurons less active.

## Observations
- synapses weights tended to saturate either max or min
- too much emergent behavior expected of the pain neurons
- pain neurons were too weak to overcome the behavior of all the other neurons.
- If pain neuron current was amplified, 
- Resolution too low for differences in spiking signal outputs (counting spikes is no good)

