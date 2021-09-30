import numpy as np
import pandas as pd

# Load data
colnames = ['x1', 'x2', 't']
train = pd.read_csv('~/Skola/Neural Networks/HW2/part2/training_set.csv', names = colnames)
validation = pd.read_csv('~/Skola/Neural Networks/HW2/part2/validation_set.csv', names = colnames)
# Normalization
def normalize(data, data_val):
    outdata = data.copy()
    outdata_val = data_val.copy()
    m1 = np.mean(data[:,0])
    m2 = np.mean(data[:,1])
    outdata[:,0] = (outdata[:,0] - m1) / np.std(data[:,0])
    outdata[:,1] = (outdata[:,1] - m2) / np.std(data[:,1])

    outdata_val[:,0] = (outdata_val[:,0] - m1) / np.std(data[:,0])
    outdata_val[:,1] = (outdata_val[:,1] - m2) / np.std(data[:,1])
    return outdata, outdata_val
train_norm, validation_norm = normalize(train.to_numpy(), validation.to_numpy())

# Initialization of weights and thresholds
def init_weights_W(M1):
    return np.random.normal(loc = 0, scale =np.sqrt(1/M1), size = (1,M1))
def init_weights_w(M1):
    return np.random.normal(loc = 0, scale = np.sqrt(0.5), size = (M1,2))
def init_thresholds(M1):
    return np.zeros((M1,1)), 0

# Local fields per mu
def calc_b(ws, x_row, thetas):
    M1 = ws.shape[0] 
    bs = np.zeros((M1,1))
    for ix in range(M1):
        bs[ix] = np.matmul(ws[ix,:], np.transpose(x_row))
    return bs
def calc_V(

