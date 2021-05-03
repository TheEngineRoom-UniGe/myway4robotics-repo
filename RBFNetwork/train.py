import numpy as np
import sys, os
import time
from lib.rbfnet import RBFN
from lib.utils import preprocess, parse_data
import json

# Parse json data passed as argument to the script
data_train = parse_data(sys.argv[1])
                  
# Pre-processing of training data
X_train, Y_train = preprocess(data_train)
                   
# Model creation
model = RBFN(hidden_shape=100, sigma=5)
tStart = time.time()

# Model training
model.fit(X_train,Y_train)
tEnd = time.time()

print("Model trained!")
print("Training time: " + str(tEnd - tStart)) 
