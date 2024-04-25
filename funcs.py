"""
    File Containing Helper functions for the code
    ispike(dt) - generates an np.array of the current spike

"""

import numpy as np
import matplotlib.pyplot as plt

def ispike(dt : float = 0.1, rt : float = 2, ft : float = 35, holdTime : float = 0):
    """
        Calculates the current response resulting from a input voltage spike
        Tuneable based on the below parameters:
            rt = rise time (in ms); 5 time constants
            ft = fall time (in ms); 5 time constants
            holdTime = time at peak (in ms)
            dur = duration (in ms); total duration of spike, rt + ft
            dt = time step (in ms)
    """
   
    #rt = 2        # Rise time in ms
    #ft = 35       # fall time in ms
    numConsts = 5 # Number of time constants
    rise_tau = rt/numConsts
    fall_tau = ft/numConsts

    dtinv = int(1/dt)
    # print(dtinv)
    # Time vector
    dur = rt + ft + holdTime
    
    trise = np.array(range(0, rt*dtinv,1)) / dtinv
    tfall = np.array(range(rt * dtinv + 1, dur * dtinv + 1, 1)) / dtinv

    a = 1 - np.exp(-1 * numConsts)
    b = a * np.exp(-1 * (ft - rt - holdTime) / fall_tau)
    # print(a)
    # print(b)
    # Calculate the first part of the spike
    irise = (1 - np.exp(-1 * trise / rise_tau ))
    ifall = (a+b)*np.exp(-1*(tfall - rt) / fall_tau)
    ihold = np.array(a * np.ones(int(holdTime/dt)))

    itotal = np.append(irise, ihold)
    itotal = np.append(itotal, ifall)
    # Normalize current
    maxI = np.max(itotal)

    t = np.append(trise, tfall)

    return {'current' : itotal/maxI,
            'time'    : t
            }

""" Tester Functions """
def _test_ispike(dt : float = 0.1):
    """
        Tester function for ispike() function
    """
    # Test code
    outputs = ispike(dt)
    current = outputs['current']
    time = outputs['time']

    print(len(current))
    print(len(time))
    print(current)
    
    # print(current)
    plt.figure()
    plt.plot(time, current, 'r-' )
    plt.xlabel("Spike Time [t] (ms)")
    plt.ylabel("Current [I] (A)")
    plt.title("Current Spike")
    plt.grid()
    plt.show()


def floatRange(start : float, end : float, delta : float):
    """
        Like range but for floats
    """
    num = int((end - start)/delta)
    y = np.empty(num)
    y[0] = start
    for i in range(1, num - 1):
        y[i] = y[i-1] + delta

    y[-1] = end

    return y

def actCompare(cur1 : list,
               spikes1 : list, 
               spikes2 : list, 
               dt : float = 0.01, 
               endTime : int = 100, 
               startTime : int = 20,
               maxDelay : int = 35) -> float:
    """
        Compares the activity of two spike series.
        INPUTS:
            spikes1 - first spike series, list of time when the spikes occurred
            spikes2 - second spike series to compare to.  List of times when the spikes occurred
            dt - simulation time step
            endTime - maximum time of series
            startTime - minium Time to start comparing at.  Model results in early instability
            maxDelay - maximum time in ms to consider spike causality
    """

    # Quantify the activity of the first spiking signal
    #act1 = actQuant1(spikes=spikes1, dt = dt, endTime = endTime, startTime = startTime)
    act1 = actQuant2(cur=cur1)

    # Get the timing difference from spikes1 -> spikes2
    timeDifs = timeComp(spikes1 = spikes1, spikes2 = spikes2, dt = dt, startTime= startTime)

    # Set the min and max number of samples for the two to be considered correlated
    minDelay = 2 / dt # 2 ms is the default rise time of the current. Seems reasonable
    maxDelay = maxDelay / dt 

    # Only counting the spikes in spikes2 that could have been caused by spike1
    numCaused = 0
    for dif in timeDifs:
        if dif >= minDelay and dif < maxDelay:
            # valid
            # Add a penalty for increased time since the spike
            numCaused= numCaused + (1 - (dif - minDelay)/maxDelay)

    # print(numCaused) #DEBUG
    # now, we've calculated how many spikes in spikes2 could have been caused by spikes in spikes1
    # Calculate the causal activation value of spikes 2
    maxSpikes = (endTime - startTime) / 4 # everything in units of ms
    if act1 == 0:
        return 0
    else:
        causeAct2 = float(numCaused / (act1 * maxSpikes))     # Causal Activity of spiking signal 2 in reference to spiking signal 1
    # If greater than 1, saturate.  This may be a problem later down the line DEBUG
    if causeAct2 > 1.0:
        causeAct2 = 1.0 # Cause act 2 is the ratio of # possibly caused spikes in the second neuron to # spikes in first

    #print(numCaused)
    #print(maxSpikes)
    #print(act1)

    # Compare the activity of spiking signal 1 to causal Activity of spikes2.
    # Calculate the squared difference between the two functions
    # Because both functions should be less than 1
    # Whole thing weights the causal activity
    # sharedActivity = causeAct2 * (1.0 - ((act1 - causeAct2) * (act1 - causeAct2)))
    # sharedActivity = causeAct2 * (1.0 - (act1 - causeAct2))
    
    return causeAct2
    #return causeAct2

def actQuant2(cur : list)-> float:
    """
        Calclate activation based on average current, NOT counting voltage spikes
    """
    return np.mean(cur)

def actQuant(spikes : list, 
             dt : float = 0.01, 
             endTime : int = 100, 
             startTime : int = 20) -> float:
    """
        Quantifies the activity of a given spike signal
        INPUTS:
            spikes - list of times when a spike occured
            dt - time step of simulation
            endTime - end time to consider
            startTime - start time to consider: spiking model results in early instability

        OUTPUTS:
            actVal - A value 0-1 quantifying neuron activation status. 
                1 means maximum spiking activity (# spikes = (endTime - startTime)/ 4ms)
                0 means no spiking activity, i.e. # spikes = 0
                any values greater than 1 will, for now, be saturated at 1
    """
    
    maxSpikes = (endTime - startTime) / 4 # everything in units of ms

    
    # Count the spikes to quantify activation
    # find the first element greater than the start index
    startInd = int(startTime/dt)
    i = 0
    while i < len(spikes) and spikes[i] < startInd:
        i = i + 1

    if i >= len(spikes):
        # no spikes found
        actVal = 0.0
    else:
        # some spikes found
        numSpikes = len(spikes[i::1])
        actVal = numSpikes / maxSpikes
        if actVal > 1:
            actVal = 1.0

    return actVal

def timeComp(spikes1 : list, 
             spikes2 : list,
             dt : float = 0.01, 
             startTime : int = 20) -> list:
    """
        Compares the timing of two spike series
        determines the index difference between each spike in spike1 and the next spike in spikes 2
        INPUTS:
            spikes1 - first spike series, list of time when the spikes occurred
            spikes2 - second spike series to compare to.  List of times when the spikes occurred
            dt - simulation time step
            startTime - minium Time to start comparing at.  Model results in early instability
        OUTPUTS:
            a vector of the same length as spikes1, containing the time to the next spike in spikes 2
            (considering causal interactions only- spikes at the same time not considered)
            (if no subsequent spike, value is listed as -1)
    """
    
    timeDifs = -1 * np.ones(len(spikes1))

    # Only consider values in spikes 2 after the start Time
    startInd = int(startTime/dt)
    spikes2Ind = 0
    while spikes2Ind < len(spikes2) and spikes2[spikes2Ind] < startInd:
        spikes2Ind = spikes2Ind + 1

    if spikes2Ind >= len(spikes2):
        # spikes 2 has no spikes of interest
        return timeDifs

    # Start filling out the timeDif vector
    spikes1Ind = 0
    while spikes1Ind < len(spikes1):
        
        # Check if there are more spikes in spikes2 to compare to
        if spikes2Ind >= len(spikes2):
            # Spikes 2 is now empty.
            return list(timeDifs)
        
        spike = spikes1[spikes1Ind]
        
        # skip spikes in starting instability interval
        if spike > startInd:
            # Out of the starting instability interval
            if spike < spikes2[spikes2Ind]:
                # There is a subsequent spike and we've found it.
                # Save the time difference
                timeDifs[spikes1Ind] = spikes2[spikes2Ind] - spike
                spikes1Ind = spikes1Ind + 1

            else:
                # There may be a subsequent spike, but we need to check and see
                spikes2Ind = spikes2Ind + 1
        else:
            spikes1Ind = spikes1Ind + 1

    return (timeDifs)



    
if __name__ == "__main__":
    _test_ispike()



