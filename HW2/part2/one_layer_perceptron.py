import pandas as pd
import numpy as np
from sklearn import preprocessing

# Load data
colnames = ['x1', 'x2', 't']
train = pd.read_csv('training_set.csv', names = colnames)
validation = pd.read_csv('validation_set.csv', names = colnames)
# Normalization
train_norm = pd.DataFrame(preprocessing.normalize(train[['x1','x2']]))
validation_norm = pd.DataFrame(preprocessing.normalize(validation[['x1','x2']]))
# Add target values and convert to numpy
train_norm['t'] = train['t']
train_norm = train_norm.to_numpy()
validation_norm['t'] = validation['t']
validation_norm = validation_norm.to_numpy()

