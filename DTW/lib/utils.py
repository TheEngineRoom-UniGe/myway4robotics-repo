import sys, os
import numpy as np
import json
import math
import collections
import random
from sklearn import preprocessing

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

# Parse json data and parameters required by the DTW algorithm
def parse_data(json):
    # Parse fullWindow
    items = json['fullWindow']['items']
    nRows = json['fullWindow']['rangeLength']
    nCols = len(items)
    fullwindow = np.zeros((nRows, nCols))
    for j in range(nCols):
        values = items[j]['values']
        for i in range(nRows):
            fullwindow[i,j] = values[i]['value']
    # parse label
    startIndex = json['label']['startIndex']
    endIndex= json['label']['endIndex']
    label = fullwindow[startIndex:endIndex]
    # Parse correspondence and overlap
    correspondence = json['correspondence']
    overlap = json['overlap']
    return fullwindow, label, correspondence, overlap

# Search for occurrencies of the given template waveform into the whole time series
# Params:
# fullWindow -> the whole time-series on which the search is run (ndarray)
# template -> the reference waveform whose replicas we're looking for in the fullWindow (ndarray)
# correspondence -> percentage level of desired correspondence between template and a possible match (int [0-100])
# overlap -> percentage value which controls the overlap between two consecutive rolling windows (int [0-99])
# Returns:
# List of starting indices of the correspondences found in fullWindow
def searchOccurrences(fullWindow, template, correspondence, overlap):
    DISTPERSAMPLE = 0.5
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
            #print(i, end="\r")
            continue
        i += increment
        #print(i, end="\r")
    return indices