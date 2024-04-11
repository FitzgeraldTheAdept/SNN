"""
    File Containing Helper functions for the code
    ispike(dt) - generates an np.array of the current spike

"""

import numpy as np
import matplotlib.pyplot as plt


def ispike(dt : float = 0.1 ):
    """
        Calculates the current response resulting from a input voltage spike
        Tuneable based on the below parameters:
            rt = rise time (in ms); 5 time constants
            ft = fall time (in ms); 5 time constants
            dur = duration (in ms); total duration of spike, rt + ft
            dt = time step (in ms)
    """
   
    rt = 2        # Rise time in ms
    ft = 35       # fall time in ms
    numConsts = 5 # Number of time constants
    rise_tau = rt/numConsts
    fall_tau = ft/numConsts

    dtinv = int(1/dt)
    # print(dtinv)
    # Time vector
    dur = rt + ft
    
    trise = np.array(range(0, rt*dtinv,1)) / dtinv
    tfall = np.array(range(rt * dtinv + 1, dur * dtinv + 1, 1)) / dtinv

    a = 1 - np.exp(-1 * numConsts)
    b = a * np.exp(-1 * (ft - rt) / fall_tau)
    # print(a)
    # print(b)
    # Calculate the first part of the spike
    irise = (1 - np.exp(-1 * trise / rise_tau ))
    ifall = (a+b)*np.exp(-1*(tfall - rt) / fall_tau)

    itotal = np.append(irise, ifall)
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


if __name__ == "__main__":
    _test_ispike()

