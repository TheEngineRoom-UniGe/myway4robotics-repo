import numpy as np
import sys
from lib.rbfnet import RBFN
from lib.utils import *
import json

# get live sensor data from the gateway
sensorData = json.loads(sys.argv[1])

with open('evaluationPy/sensorsConfig.json', "r") as file:
    # Reading from file
    cloudData = json.loads(file.read())

# Load model from previous training
model = load_model_dict(cloudData)
# Load evaluation parameters
sw_size, threshold = get_evaluation_params(cloudData)

# Parse data from json
data = parse_data(sensorData)
# Preprocessing of test data
X, Y = preprocess(data)

sw = []
for i in range(len(data)):
    # Predict acceleration at next time instant
    yp = model.predict([X[i]]).reshape(-1)
    # Compute RMSE between predicted and measured accelerations
    r = rmse(yp, Y[i])
    # Append current RMSE value to sliding window
    if (len(sw) >= sw_size):
        del sw[0]
    sw.append(r)
    # if average RMSE in the sw is greater than threshold, fire alarm
    if (np.mean(sw) > threshold):
        sensorData[i]['Failure'] = 1
    else:
        sensorData[i]['Failure'] = 0


# deca: convert single quotes in double quotes for sending a valid json to iothub
print(json.dumps(sensorData))
sys.stdout.flush()
