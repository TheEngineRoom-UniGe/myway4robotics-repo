import numpy as np
import pandas as pd
import serial
import sys
sys.path.insert(1, '')
from rbfnet_oneshot import RBFN

# Define rmse function
def rmse(targets, predictions):
    return np.sqrt(np.mean((predictions-targets)**2))
    
data = pd.read_csv('/home/simone/Desktop/MyWay4Robotics/Test/train_trajectory.csv',
                   sep=',', 
                   header=None,
                   dtype=np.float, 
                   na_values="Null").values[1:]


X_train = data
Xtn = data[1:, :]
Y_train = np.vstack([Xtn, list(Xtn[-1])])


model = RBFN(hidden_shape=100, sigma=5)
print("Offline model training...")
model.fit(X_train,Y_train)
print("Model trained and ready!\n")

ids = ['e3a4', 'd80e']
id_to_idx = {'e3a4': 0, 'd80e': 1}
dim = len(ids) * 3
row_filled = [0] * int(dim / 3)
new_row = [0] * dim
step = 0.02
end = 19
hasFirst = False
sw = []
realT = 0
ser = serial.Serial("/dev/ttyACM1", timeout=1)
input("Press Enter to initialize joint monitoring routine...")
print("Routine started!\n")
while realT < end:
    line = ser.readline().decode('utf-8').split(' ')
    sensor_id = line[0]
    if (len(line) > 3 and sensor_id in ids):

        acc_x = float(line[1])
        acc_y = float(line[2])
        acc_z = float(line[3])

        idx = id_to_idx[sensor_id]
        cols = idx * 3

        new_row[cols:cols + 3] = [acc_x, acc_y, acc_z]

        row_filled[idx] = 1
        # If new data vector is available
        if (sum(row_filled) == 2):

            if (not hasFirst):
                hasFirst = True
                datap = new_row

            else:
                # Perform prediction at current time instant
                yp = model.predict([datap]).reshape(-1)
                r = rmse(yp, np.array(new_row))
                if (len(sw) >= 20):
                    del sw[0]
                sw.append(r)
                if (np.mean(sw) > 0.0625):
                    print("Fault Detection:  Joint 3", end='\r')
                else:
                    print("Fault Detection: No Fault", end='\r')

                # Reset to receive next data vector
                realT += step
                datap = new_row
                row_filled = [0] * int(dim / 3)
                new_row = [0] * dim
print("\n")
print("Done")
