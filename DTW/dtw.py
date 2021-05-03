import numpy as np
import json
import math
import collections
import random
from sklearn import preprocessing

DISTPERSAMPLE = 0.5

# DTW algorithm implementation
# Params:
# s,t -> time series of data organized as numpy ndarray
# returns:
# Distance value (float) between the two time series
def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0
        
    for i in range(1, n+1):
        for j in range(np.max([1, i-1]), np.min([m, i+1])+1):
            dtw_matrix[i, j] = 0
            cost = np.linalg.norm(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[-1,-1]

# Parse json data into numpy ndarray required by the algorithm
def parse_data(json):
    n = json['values']['length']
    items = json['values']['items']
    data = np.zeros((n, 1))
    for i, detection in enumerate(items):
        data[i, 0] = detection['accelerationx']
    return data

# Search for occurrencies of the given template waveform into the whole time series
# Params:
# fullWindow -> the whole time-series on which the search is run (ndarray)
# template -> the reference waveform whose replicas we're looking for in the fullWindow (ndarray)
# correspondence -> percentage level of desired correspondence between template and a possible match (int [0-100])
# overlap -> percentage value which controls the overlap between two consecutive rolling windows (int [0-99])
# Returns:
# List of starting indices of the correspondences found in fullWindow
def searchOccurrences(fullWindow, template, correspondence, overlap):
    tempLen = len(template)
    # Compute threshold based on required level of correspondence
    maxDist = DISTPERSAMPLE * tempLen
    threshold =  maxDist - (maxDist / 100) * correspondence
    # Compute increment to guarantee fixed overlapping
    increment = math.ceil(tempLen*(1 - overlap / 100))
    if(increment <= 0):
        increment = 1
    
    indices = []
    i = 0
    while i < len(fullWindow) - tempLen:
        currentDistance = dtw(template, fullWindow[i:i+tempLen])
        if(currentDistance <= threshold):
            indices.append(i)
            i += tempLen
            print(i, end="\r")
            continue
        i += increment
        print(i, end="\r")
    return indices


if __name__ == "__main__":

    # TO DO: modify in order to accept the parameters as arguments
    # sensorData = sys.argv[1]
    # template = sys.argv[2]
    # correspondence = sys.argv[3]
    # overlap = sys.argv[4]

    # Test execution with ecg data
    with open('ecg.json', "r") as file:
        sensorData = json.loads(file.read())

    data = parse_data(sensorData)
    # Data needs to be normalized in range [0,1] in order for the DTW algorithm to be effective
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    start = 700
    end = 950
    waveRef = data[start:end]

    indices = searchOccurrences(data, waveRef, 90, 98)
    print(indices)