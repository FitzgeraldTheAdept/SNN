# Generate all the data needed for this code
# Sort into 3 categories:
#   Training
#   Testing
#   Validation
#
# Training Data is 4 pixel square images.  Effectively it's just four values 0-100 in a list here
# 
# example_image = [A, B, C, D] => A B
#                                 C D

import json
import funcs
import numpy as np

_CROSS = 1  # type code for crosses
_VBAR = 2   # type code for vertical bars
_HBAR = 3   # type code for horizontal bars


offThresh = 25 # Maximum brightness value to be considered off
onThresh = 50 # Minimum brightness value to be considered on
# Anything between offThresh and onThresh is illegal

def buildImg(A : int, B : int, C : int, D : int) -> list:
    """
        Build all the images for the generic image A B C D
        value of 1 indicates pixel is on
        value of 0 indicates pixel is off
    """
    if A == 1:
        Alim = [onThresh, 100]
    else:
        Alim = [0, offThresh]

    if B == 1:
        Blim = [onThresh, 100]
    else:
        Blim = [0, offThresh]

    if C == 1:
        Clim = [onThresh, 100]
    else:
        Clim = [0, offThresh]

    if D == 1:
        Dlim = [onThresh, 100]
    else:
        Dlim = [0, offThresh]
    imgs = list()
    testRes = 2
    for A in range(Alim[0], Alim[1], testRes):
        for B in range(Blim[0], Blim[1], testRes):
            for C in range(Clim[0], Clim[1], testRes):
                for D in range(Dlim[0], Dlim[1], testRes):
                    imgs.append([int(A), int(B), int(C), int(D)])

    return imgs


def buildCrosses() -> list:
    """
        Generates the Crosses
        Crosses have shape      1 0     or      0 1
                                0 1             1 0
    """
    # Start with generating the NW-SE crosses
    NWSECrosses = buildImg(A=1, B=0, C=0, D=1)
    print("Built NWSE Crosses")

    # Then generate the NE-SW crosses
    NESWCrosses = buildImg(A=0, B=1, C=1, D=0)
    print("Built NESW Crosses")

    # Combine the lists
    #data = dict()
    #data['type'] = "Crosses"
    #data['images'] 
    crosses = NWSECrosses + NESWCrosses

    return crosses


def buildVertBars() -> list:
    """
        Generates the Vertical Bars
        Vertical Bars have the shape    1 0     or      0 1
                                        1 0             0 1
    """

    # Start with generating Left Vertical Bars
    LeftVBars = buildImg(A = 1, B = 0, C = 1, D = 0)
    print("Built Left Vertical Bars")

    # Then the right
    RightVBars = buildImg(A = 0, B = 1, C = 0, D = 1)
    print("Built Right Vertical Bars")

    # combine
    vBars = LeftVBars + RightVBars

    return vBars

def buildHorzBars() -> list:
    """
        Generates the Horizontal Bars
        Horizontal Bars have the shape:     1 1     or      0 0
                                            0 0             1 1
    """
    # Start with generating top horizontal bars
    topHBars = buildImg(A=1, B=1, C=0, D=0)
    print("Built Top Horizontal Bars")

    # Generate the bottom horizontal Bars
    botHBars = buildImg(A=0, B=0, C=1, D=1)
    print("Built Bottom Horizontal Bars")

    # combine
    hBars = topHBars + botHBars

    return hBars

def buildSets(train : list, test : list, valid : list, fromList : list, type : int):
    """
        Build the train, test, and validation sets
        INPUTS:
            fromList - the set to take from (i.e. crosses, vBars, hBars)
            type     - the type of image this is
        OUTPUTS: 
            Nothing, but the training, testing, and validation lists will have all the things added
    """
    i = 0
    for img in fromList:
        if i < 3:
            # Add 3 to training
            train.append([img, type])
            i = i + 1

        elif i == 3:
            # Add 1 to testing
            test.append([img, type])
            i = i + 1
        else:
            # Add 1 to validation
            valid.append([img, type])
            i = 0

    
if __name__ == "__main__":
    crosses = buildCrosses()
    print("Built Crosses")
    vBars = buildVertBars()
    print("Built Vertical Bars")
    hBars = buildHorzBars()
    print("Built Horizontal Bars")

    
    # Write to JSON file
    path = './data/'
    """
    # Create output dictionary for json file
    allImgs = dict()
    allImgs['Num Crosses'] = len(crosses)
    allImgs['Crosses'] = crosses
    allImgs['Num VBars'] = len(vBars)
    allImgs['VBars'] = vBars
    allImgs['Num HBars'] = len(hBars)
    allImgs['HBars'] = hBars

    
   
    # write to file
    with open('{}{}.json'.format(path,"allImages"), 'w') as f:
        json.dump(allImgs, f)
    """

    # Create validation, testing, and training datasets
    # 60, 20, 20 split
    # in other words, throw 3 random values in training, then 1 in testing, then 1 in validation

    testing = list()
    training = list()
    validation = list()

    # Add the stuff to the sets
    # Importantly, this list will be sorted by category.  It's important to access this randomly when actually using
    buildSets(train=training, test=testing, valid=validation, fromList=crosses, type=_CROSS)
    print("Built Cross set")
    buildSets(train=training, test=testing, valid=validation, fromList=vBars, type=_VBAR)
    print("Built Vertical Bar set")
    buildSets(train=training, test=testing, valid=validation, fromList=hBars, type=_HBAR)
    print("Built Horizontal Bar set")

    # Create dictionaries
    testDict = dict()
    trainDict = dict()
    validDict = dict()

    trainDict['Num Images'] = len(training)
    testDict['Num Images'] = len(testing)
    validDict['Num Images'] = len(validation)

    trainDict['Images'] = training
    testDict['Images'] = testing
    validDict['Images'] = validation

    # Print Training data to file
    with open('{}{}.json'.format(path,"train"), 'w') as f:
        json.dump(trainDict, f)

    print('Wrote to {}{}.json'.format(path,"train"))

     # Print Testing data to file
    with open('{}{}.json'.format(path,"test"), 'w') as f:
        json.dump(testDict, f)
    
    print('Wrote to {}{}.json'.format(path,"test"))

     # Print Validation data to file
    with open('{}{}.json'.format(path,"valid"), 'w') as f:
        json.dump(validDict, f)

    print('Wrote to {}{}.json'.format(path,"valid"))