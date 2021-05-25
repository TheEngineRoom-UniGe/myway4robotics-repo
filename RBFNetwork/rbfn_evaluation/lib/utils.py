import numpy as np
import json
from lib.rbfnet import RBFN

def parse_data(json):
    # Parse data and detection times
    sensorId = json['window']['sensorId'] 
    items = json['window']['items']
    nRows = json['window']['rangeLength'] 
    nCols = len(items)
    data = np.zeros((nRows, nCols))
    detectionTimes = np.zeros((nRows,1))
    for j in range(nCols):
        values = items[j]['values']
        for i in range(nRows):
            data[i,j] = values[i]['value']
            if j == 0:
                detectionTimes[i] = values[i]['detectiontime']
    # Get max and min values
    maxV = json['sensorMaxValue']
    minV = json['sensorMinValue']
    # Parse model dictionary and instantiate model
    modelDict = json['modelDict']
    h_shape = modelDict['hiddenShape']
    sigma = modelDict['sigma']
    model = RBFN(hidden_shape=h_shape, sigma=sigma)
    model.centers = np.array(modelDict['centers'])
    model.weights = np.array(modelDict['weights'])
    # Parse evaluation parameters
    sw = json['slidingWindow']
    threshold = json['threshold']
    return data, detectionTimes, sensorId, maxV, minV, model, sw, threshold

def preprocess(data, maxV, minV):
    nCols = data.shape[1]
    maxValue = np.array(nCols*[maxV])
    minValue = np.array(nCols*[minV])
    data_scaled = (data - minValue) / (maxValue - minValue)
    X_next = data_scaled[1:, :]
    Y = np.vstack([X_next, list(X_next[-1])])
    return data_scaled, Y

def rmse(targets, predictions):
    return np.sqrt(np.mean((predictions-targets)**2))