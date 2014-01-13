import json
from numpy import *

class DataWrapper:
    def __init__(self, filename):
        
        if (filename.endswith('.json')):
            raise NotImplementedError
            
        else:
            raise NotImplementedError
    
    # Returns a 1D array of all training frequencies (ie. SSVEP LED flashing rate) within the data
    def GetFrequencies(self):
        raise NotImplementedError
        
    # Returns a 2D array of all FFT data of the given training frequency
    #   # of rows = (# of trials run) * (# of classes) * (# of time windows FFT'ed)
    #   # of columns = (# of training frequencies)
    def GetFrequency(self, freq):
        raise NotImplementedError