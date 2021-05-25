import sys, os
import numpy as np
import json
import math
import collections
import random
from sklearn import preprocessing
from lib.utils import parse_data, searchOccurrences

if __name__ == "__main__":

    sensorData = None
    if len(sys.argv) > 1:
        sensorData = json.loads(sys.argv[1])
    
    if not sensorData:
        sys.exit('Missing json passed as argument!')
    
    '''
    with open('data/ecg_dtw.json', "r") as file:
        # Reading from file 
        sensorData = json.loads(file.read())
    '''
    
    # Parse parameters from the json file
    fullwindow, template, correspondence, overlap = parse_data(sensorData)
    # Data needs to be normalized in range [0,1] in order for the DTW algorithm to be effective
    min_max_scaler = preprocessing.MinMaxScaler()
    fullwindow_scaled = min_max_scaler.fit_transform(fullwindow)
    template_scaled = min_max_scaler.transform(template)

    indices = searchOccurrences(fullwindow_scaled, template_scaled, correspondence, overlap)
    print(json.dumps(indices))