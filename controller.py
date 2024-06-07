_OPTIONS = ["Get <info> on the network", "<Load> a network", "<Build> a network",
            "<Write> the active network", "<Train> the active network", 
            "Run <infer>ence on the active network",
            "See a <demo> of the active network", "<Plot> most recent outputs", "<Exit>"]


from trainer import Trainer
from network import Network
import matplotlib.pyplot as plt
import json
import random as rng
import numpy as np

class Controller(object):
    """ COMMANDER FILE FOR PROJECT 
    - Lets you load, write, and build network
    - Allows running inference on a loaded network
    - allows starting training on the network
    """
    def __init__(self):
        self.path = "big"
        self.toPath = self.path + "1"
        self.trainer = None
        self.network = None
        
        # start interface
        self._UI()


    def _UI(self):
        """
            Text-based user interface for controlling the network
        """
        
        print("#################################\n" + 
              "Spiking Neural Network Simulator\n" + 
              "   Final Project for EE5900\n" + 
              " Copyright Benjamin Wittrup 2024\n" + 
              "#################################\n\n")
        
        choice = -1
        
        while choice != -100:
            opts = self._getOptions()
            print("What would you like to do?\n")
            for opt in opts:
                print(_OPTIONS[opt])
            
            print(_OPTIONS[-1])
            
            # collect user input
            (choice, arg) = self._condInput(input(">>"))
            
            if choice >= 0:
                
                if choice > len(opts):
                    print("Option Not available.  Please <load> a network first.")
                elif choice == 0:
                    # Get Info
                    self._getInfo(arg)
                elif choice == 1:
                    print("Loading a Network")
                    self.path = input("Please specify file name and path, NOT including the .net extension: ")
                    try:
                        self.network = Network(self.path + ".net")
                        print(f"Network {self.path} loaded successfully.")
                    except FileNotFoundError as e:
                        print(f"Could not locate file {self.path}.net")

                elif choice == 2:
                    # Build Network
                    self._buildNet()
                elif choice == 3:
                    print("Writing the network to file")
                    self.toPath = input("Please specify file name and path, NOT including the .net extension: ")
                    try:
                        self.network.writeNetwork(path= (self.toPath + ".net"))
                    except Exception as e:
                        print(f"Could not write file to path {self.toPath}.net")
                
                elif choice == 4:
                    print("Initializing the training of active network")
                    
                    numGens = -1
                    endRoutine = 0
                    while numGens <= 0:
                        val = input("Please specify the number of generations to train for.\n>> ")
                        if val.find("exit") != -1 or val.find("end") != -1:
                            endRoutine = 1
                        elif val == '':
                            # no input specified
                            numGens = 20
                        else:
                            try:
                                numGens = int(val)
                                if numGens <= 0:
                                    print("Number of Generations must be greater than 0")
                            except ValueError as e:
                                print(f"Not a valid number.  Please input an integer")

                    numEpochs = -1
                    while not endRoutine and numEpochs <= 0:
                        val = input("Please specify the number of Epochs to train for.\n>> ")
                        if val.find("exit") != -1 or val.find("end") != -1:
                            endRoutine = 1
                        elif val == '':
                            # no input specified
                            numEpochs = 10
                        else:
                            try:
                                numEpochs = int(val)
                                if numEpochs <= 0:
                                    print("Number of Epochs must be greater than 0")
                            except ValueError as e:
                                print(f"Not a valid number.  Please input an integer")
                        
                    if not endRoutine:
                        trainerBot = Trainer(network=self.network, numGens=numGens, numEpochs = numEpochs, learnRate = 0.2)
                        
                        print("Contstructed Trainer.  Beginning Training")
                        trainerBot.trainNetwork()
                        plt.figure()
                        plt.plot(trainerBot.error)
                        plt.xlabel("Epoch")
                        plt.ylabel("MSE")
                        plt.show()

                elif choice == 5:
                    print("Running Inference on active network.")
                    print("Please specify the input pixel values (0.0 -> 1.0):")
                    pixels = list()
                    i = 0
                    while i < self.network.structure[0]:
                        #print(self.network.structure[0]) #DEBUG
                        val = input(f"Input [{i}]: ")
                        
                        if val.find("exit") != -1:
                            break
                        try:
                            pixels.append(float(val))
                            i = i + 1
                        except ValueError as e:
                            print("Invalid pixel value.  Must be between 0 and 1")

                    # no pain input current, but pixel input current
                    self.network.phase(I_in = pixels, I_pain = None)
                    print("Network outputs are: ")
                    print(self.network.getOuts())

                    input("\nPress Enter to Continue.")
                    print("\n\n")
                elif choice == 6:
                    
                    print("Starting Validation")
                    self._valid()
                elif choice == 7:
                    self.network.plotOuts()
            elif choice == -100:
                break

        
    def _condInput(self, data : str) -> tuple:
        """
            Conditions user input
            INPUTS:
                data - string from user

            Determines if you'd like to:
                get info on the network (0)
                load a network (1)
                build a network
                write the active network to file (3)
                train the active network  (4)
                run inference on the active network (5)
                see a demo of the active network (6)
                plot the outputs of the last simulation (7)
                exit (-100)

                returns -1 if value was invalid
        """
        data = data.lower()
        args = data.split(' ')
        if len(args) > 1:
            arg = args[1]

        else:
            arg = ' '

        if data.find("exit") != -1 or data.find("end") != -1:
            return (-100, ' ')
        elif data.find("info") != -1:
            return (0,arg)
        elif data.find("load") != -1:
            return (1, arg)
        elif data.find("build") != -1:
            return (2, arg)
        elif data.find("write") != -1 or data.find("save") != -1:
            return (3, arg)
        elif data.find("train") != -1:
            return (4, arg)
        elif data.find("infer") != -1 or data.find("test") != -1:
            return (5, arg)
        elif data.find("demo") != -1:
            return (6, arg)
        elif data.find("plot") != -1:
            return (7, arg)
        else:
            print(f"'{data}' command not recognized.\n")
            return (-1, ' ')
            

    def _getOptions(self) -> list:
        """
            Determines options available to the user based on state
            if no network is initialized, options are very limited
        """

        if self.network is not None:
            options = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            options = [0, 1, 2]
            
        return options


    def _valid(self):
        """
            Validation on the dataset
        """
        try:
            f = open("data/valid.json")
        except Exception as e:
            raise e

        data = json.load(f)
        f.close()
        numImgs = data['Num Images']
        imgs = data['Images']
        rng.seed(52)

       
        # get 4 images
        fig, axs = plt.subplots(nrows=2,ncols=2)
        for i in [0, 1]:
            for j in [0, 1]:
                imgInd = int(numImgs * rng.random() - 1)
                
                testVal = 1 - (np.asarray(imgs[imgInd][0]) / 100)
                self.network.phase(I_in=testVal)
                outs = self.network.getOuts()
                axs[i][j].imshow(np.resize(testVal, (2,2)), cmap=plt.get_cmap('Greys'), vmin=0, vmax=1)
               
                maxVal = -1
                maxType= -1
                z = 0
                for val in outs:
                    if val > maxVal:
                        maxVal = val
                        maxType = z
                    z = z + 1

                axs[i][j].set_title(f"{maxType} : " + ("%.3f" % maxVal))
                print(f"Checked validation: {2*i+j}")
                       
        
        plt.show()

    def _getInfo(self, arg : str):
        """
            Code for get info option
        """
        # get info
        if self.network is None and arg == ' ':
            print("Made by Benjamin Wittrup\n" + 
                    "  April 26, 2024  ")
        elif self.network is not None and arg == ' ':
            print(f"###########{self.path}###########")
            self.network.dumpInfo()
        
        else: 
            # No network loaded.
            #print("Loading a Network")
            tempPath = arg
            try:
                tempNet = Network("nets/" + tempPath + ".net")
                print(f"Network {tempPath} retrieved from file.")
                print(f"###########{tempPath}###########")
                tempNet.dumpInfo()
                opt = input(f"Would you like to switch the active network to {tempPath}? (y/n) ")
                if opt.find("y") != -1:
                    self.network = tempNet
                    self.path = tempPath
                    print(f"Network {self.path} is now active.")

            except FileNotFoundError as e:
                print(f"Could not locate file {tempPath}.net")

        
        input("\nPress enter to continue.")

    
    def _buildNet(self):
        print("Building a network.")
        structure = list()
        structure.append(self._loopUntil(castTo=int, prompt="Specify # Inputs: "))
        if structure[-1] == -100:
            # Exit value
            return
        structure.append(self._loopUntil(castTo=int, prompt="Specify # Feedback Neurons: "))
        if structure[-1] == -100:
            # Exit value
            return
        structure.append(self._loopUntil(castTo=int, prompt="Specify # Outputs: "))
        if structure[-1] == -100:
            # Exit value
            return
        
        ans = 1
        while ans > 0:
            try:
                ans = int(input("Specify # Neurons in next hidden layer: "))
                structure.append(ans)
            except Exception as e:
                ans = -100

        pD = 300
        dt = 0.01
        maxI = (110/(4 * dt))  # (v_thresh - v_reset) / (4 ms / dt)
        
        try:
            pD = int(input("Specify phase duration in ms (int): "))
        except Exception as e:
            print(f"Default value of {pD} ms will be used.")
            
        try:
            dt = float(input("Specify time step, dt in ms (float): "))
        except Exception as e:
            print(f"Default value of {dt} will be used.")

        try:
            maxI = float(input("Specify Maximum Current Amplitude: "))
        except Exception as e:
            print(f"Default value of {maxI} will be used.")

        # Build the network now
        self.network = Network(structure=structure, 
                                phaseDuration = pD,
                                dt=dt,
                                maxI = maxI)
        
        print("Network Built Successfully.")
        input("Please press Enter to Continue.")
        print("\n\n")

    def _loopUntil(self, castTo : type, prompt : str):
        """
            Handles input looping to condition user input until it can be successfully casted to the type castTo
        """
        leave = 0
        while leave == 0:
            userInput = input(prompt)
            if userInput.find("exit") != -1 or userInput.find("end") != -1:
                # Error/exit value
                return -100
            
            else:
                try:
                    val = castTo(userInput)
                    leave = 1
                except ValueError as e:
                    print(f"Could not interpret {userInput} as {castTo}.")
        
        return val


if __name__ == "__main__":
    Controller()
    pass