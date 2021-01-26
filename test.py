import numpy as np
import pandas as pd
import serial
import sys, os
import time
import pickle

# Import RBF network from separate script
sys.path.append(os.path.join(sys.path[0], 'lib'))
from rbfnet import RBFN

# Define rmse function
def rmse(targets, predictions):
    return np.sqrt(np.mean((predictions-targets)**2))
    
 # Load model parameters from file
def load_model_dict(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
        
# Load model from previous training
model_dict = load_model_dict("model.pkl")
model = RBFN(hidden_shape=model_dict['hidden_shape'], sigma=model_dict['sigma'])
model.centers = model_dict['centers']
model.weights = model_dict['weights']

"""
 Here the routine for monitoring joints status is launched. In the final application, connection to the sensors and acquisition of data
 in real-time is required. For the moment, a pre-recorded test trajectory is used
 
 Parameters to be modified/tuned on demand:
    - sw_size: Size of the sliding window considered for averaging consecutive RMSEs
    - threshold: Threshold parameter for signaling an alarm
"""
input("Press Enter to initialize joint monitoring routine...")

sw_size = 20
threshold = 0.1

data_test = pd.read_csv(os.path.join(sys.path[0], 'data')+"/test_trajectory2.csv",
                   sep=',', 
                   header=None,
                   dtype=np.float, 
                   na_values="Null").values[1:]
 
# Preprocessing of test data
X_test = data_test
Xttn = data_test[1:, :]
Y_test = np.vstack([Xttn, list(Xttn[-1])])

print("Routine started!\n")
tStart = time.time()

i = 0
sw = []

while i < len(X_test):
    # Predict acceleration at next time instant
    yp = model.predict([X_test[i]]).reshape(-1)
    # Compute RMSE between predicted and measured accelerations
    r = rmse(yp, Y_test[i])
    # Append current RMSE value to sliding window
    if (len(sw) >= sw_size):
        del sw[0]
    sw.append(r)
    # if average RMSE in the sw is greater than threshold, fire alarm
    if (np.mean(sw) > threshold):
        """
            Possibility of sending a timestamp corresponding to the beginning of the alarm.
        """
        print("Fault Detection:  Joint X", end='\r')
    else:
        print("Fault Detection: No Fault", end='\r')
    i += 1
    time.sleep(0.02)
print("\n")
tEnd = time.time()
print("Joint monitoring routine time: " + str(tEnd - tStart)) 
