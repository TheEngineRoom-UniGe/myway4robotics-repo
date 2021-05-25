import numpy as np
from lib.rbfnet import RBFN
import lib.utils as utilities
import sys
import json
import os
import time


if __name__ == "__main__":

    '''
    sensorData = None
    if len(sys.argv) > 1:
        sensorData = json.loads(sys.argv[1])
    
    if not sensorData:
        sys.exit('Missing json passed as argument!')
    '''
    
    with open('data/ecg_detections.json', "r") as file:
        # Reading from file 
        sensorData = json.loads(file.read()) 
    

    # Extract data and training parameters from the input json 
    data, detectionTimes, sensorId, maxV, minV, model, sw_size, threshold = utilities.parse_data(sensorData)
    # Preprocess and normalize data
    X, Y = utilities.preprocess(data, maxV, minV)

    failures_list = []
    sw = []
    for i in range(len(data)):
        # Predict acceleration at next time instant
        yp = model.predict([X[i]]).reshape(-1)
        # Compute RMSE between predicted and measured accelerations
        r = utilities.rmse(yp, Y[i])
        # Append current RMSE value to sliding window
        if (len(sw) >= sw_size):
            del sw[0]
        sw.append(r)
        # if average RMSE in the sw is greater than threshold, fire alarm
        if (np.mean(sw) > threshold):
            # Get second corresponding to the timestamp
            second = int(detectionTimes[i]/1000)
            last_idx = len(failures_list) - 1
            # If no failure yet reported or failure belongs to another second
            if (not failures_list or second != failures_list[last_idx]['timestamp']):
                failures_list.append({"sensorId": sensorId, 
                                    "timestamp": second, 
                                    "failureStart": i, 
                                    "failureEnd": i}
                )
            # Otherwise, if the failure bolongs to the current second, extend current failure end
            elif (second == failures_list[last_idx]['timestamp']):
                failures_list[last_idx]['failureEnd'] = i
            else:
                continue        
        else:
            continue

    print(json.dumps(failures_list))