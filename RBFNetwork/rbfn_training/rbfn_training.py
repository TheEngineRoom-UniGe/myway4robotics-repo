import numpy as np
import sys
import json

# RBFN class definition

class RBFN(object):

    def __init__(self, hidden_shape, sigma=1.0):
        """ radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: the number
            hidden_shape: number of hidden radial basis functions,
            also, number of centers.
        """
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(center-data_point)**2)

    def _calculate_interpolation_matrix(self, X):
        """ Calculates interpolation matrix using a kernel_function
        # Arguments
            X: Training data
        # Input shape
            (num_data_samples, input_shape)
        # Returns
            G: Interpolation matrix
        """
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(
                        center, data_point)
        return G

    def _select_centers(self, X):
        random_args = np.random.choice(len(X), self.hidden_shape, replace=False) 
        centers = X[random_args]
        return centers

        
    def fit(self, X, Y):
        """ Fits weights using linear regression
        # Arguments
            X: training samples
            Y: targets
        # Input shape
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        self.centers = self._select_centers(X)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions

# Utility methods 

def parse_training_data(json):
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
    hiddenShape = json['hiddenShape']
    sigma = json['sigma']
    return data, maxV, minV, hiddenShape, sigma

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


if __name__ == "__main__":

    
    inputData = None
    if len(sys.argv) > 1:
        inputData = json.loads(sys.argv[1])
    
    if not inputData:
        sys.exit('Missing json passed as argument!')
    
    '''
    with open('data/ecg_training_input.json', "r") as file:
        # Reading from file 
        inputData = json.loads(file.read())
    '''
    
    # Parse data from input Json file
    data_train, maxV, minV, hiddenShape, sigma = parse_training_data(inputData)
    # Preprocess and normalize training data
    X_train, Y_train = preprocess(data_train, maxV, minV) 

    # Model creation and training
    model = RBFN(hiddenShape, sigma)
    model.fit(X_train, Y_train)

    # Model dictionary
    modelDict = {}
    modelDict['hiddenShape'] = model.hidden_shape
    modelDict['sigma'] = model.sigma
    modelDict['centers'] = model.centers.tolist()
    modelDict['weights'] = model.weights.tolist()
    print(json.dumps(modelDict))