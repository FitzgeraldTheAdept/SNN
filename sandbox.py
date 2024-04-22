"""
Test file (for running random code bits before chucking into the rest of the project)
"""

# importing libraries
from synapse import Synapse
from timeit import default_timer as timer
import itertools
import random
from neuron import Neuron
import numpy as np

# Function under evaluation
def test_func(test_set):
	# x = lambda val: val
	x = lambda val: other_func(val)
	return map(x, test_set)
	#for val in test_set:
	#	_ = val
 
def test_func2(test_set):
	x = list(test_set)
	y = np.add(x, 1)
	return y
		

def other_func(val : int):  
	return val + 1
	


def timeTest():
	random.seed(21)
	for _ in range(5):
		test_set = set()

		# generating a set of random numbers
		for el in range(int(1e6)):
			el = random.random()
			test_set.add(el)

		start = timer()
		out = list(test_func(test_set))
		end = timer()


		print(str(end - start))

		start = timer()
		out = test_func2(test_set)
		end = timer()
		print(str(end-start))
		# print(out[0])
		
# Driver function
if __name__ == '__main__':
    #timeTest()

	from matplotlib import colormaps
	print(list(colormaps))
	



    #print(test[1].params['c'])
	
    #print(len(test))
