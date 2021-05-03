import numpy as np
from rbfnet import RBFN

def load_model_dict(json_data):
    model_dict =  json_data['TrainedData']['ModelDict']
    hidden_shape = model_dict['HiddenShape']
    sigma = model_dict['Sigma']
    centers = np.array(model_dict['Centers'])
    weights = np.array(model_dict['Weights'])
    model = RBFN(hidden_shape=hidden_shape, sigma=sigma)
    model.centers = centers
    model.weights = weights
    return model

def get_evaluation_params(json_data):
    params = json_data['TrainedData']['AlgorithmParameters']
    sliding_window = int(params[0]['Value'])
    threshold = float(params[1]['Value']) / 100
    return sliding_window, threshold

def parse_data(json):
    data = np.zeros((len(json), 3))
    for i, detection in enumerate(json):
        data[i, 0] = detection['Accelerationx']
        data[i, 1] = detection['Accelerationy']
        data[i, 2] = detection['Accelerationz']
    return data

def preprocess(data):
    X_next = data[1:, :]
    Y = np.vstack([X_next, list(X_next[-1])])
    return data, Y

def rmse(targets, predictions):
    return np.sqrt(np.mean((predictions-targets)**2))
