import numpy as np
import pandas as pd
import serial
import sys, os
import time


"""
 This script can be invoked by command line similarly to the following example:
 
 python train.py 100 5 "0.1,0.5,-0.3;0.2,-0.3,0.5"
 
 - The first argument corresponds to hidden_shape, the parameter controlling the size of the 
 hidden layer
 - The second argument is sigma, the parameter which controls the level of non-linearity of
 the network
 - The third argument represents the training data. The data must be encapsulated in " "
 brackets, values of each row separated by a comma and rows must be separated by semicolon.
 For example "0.1,0.5,-0.3;0.2,-0.3,0.5" corresponds to two consecutive readings of a
 single sensor. In case of five sensors, one row will be composed of 15 measurements instead
 of 3.
 
 The script outputs a trained model saved as .pkl format
 
"""

# Import RBF network from separate script
sys.path.append(os.path.join(sys.path[0], 'lib'))
from rbfnet import RBFN
    
hidden_shape = int(sys.argv[1])
sigma = int(sys.argv[2])

data_train_list = sys.argv[3].replace(",", " ")
data_train = np.squeeze(np.asarray(np.matrix(data_train_list)))
                   
# Pre-processing of training data
X_train = data_train
Xtn = data_train[1:, :]
Y_train = np.vstack([Xtn, list(Xtn[-1])])
   
# Model creation
model = RBFN(hidden_shape=hidden_shape, sigma=sigma)
tStart = time.time()

# Model training
model.fit(X_train,Y_train)
tEnd = time.time()
# Save model for future use
model.save()
print("Model trained!")
print("Training time: " + str(tEnd - tStart)) 
