import numpy as np
import pandas as pd
import serial
import sys, os
import time

# Import RBF network from separate script
sys.path.append(os.path.join(sys.path[0], 'lib'))
from rbfnet import RBFN
    
"""    
 Load training dataset from previously collected CSV.
 Here the final application will connect to the stream of sensors, send commands to 
 the robot in order to start the movement and collect the corresponding data for 
 training purpose.
 
  Parameters to be modified/tuned on demand:
    - hidden_shape: Size of the RBF's hidden layer
    - sigma: parameter controlling the level of non-linearity of the network
 
"""
data_train = pd.read_csv(os.path.join(sys.path[0], 'data')+"/train_trajectory.csv",
                   sep=',', 
                   header=None,
                   dtype=np.float, 
                   na_values="Null").values[1:]
                   
# Pre-processing of training data
X_train = data_train
Xtn = data_train[1:, :]
Y_train = np.vstack([Xtn, list(Xtn[-1])])
                   
# Model creation
model = RBFN(hidden_shape=100, sigma=5)
tStart = time.time()

# Model training
model.fit(X_train,Y_train)
tEnd = time.time()
# Save model for future use
model.save()
print("Model trained!")
print("Training time: " + str(tEnd - tStart)) 
