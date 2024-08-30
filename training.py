import wfdb
import os
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import pywt
import spkit as sp
from functions import smooth, Fourier, SNR, moving_average, remove_close_values, features

data_dir = '/Users/v.w./Desktop/intern/research/2024_summer/hkust/mit-bih-arrhythmia-database'
files = os.listdir(data_dir)
record = wfdb.rdrecord(os.path.join(data_dir, '100'))
annotation = wfdb.rdann(os.path.join(data_dir, '100'), 'atr')
signal = record.p_signal


plot_a = signal[:2500, 0]

print(features(plot_a))

# Convert the features and labels to numpy arrays
X = np.array(features(signal[:i]) for i in range(len(signal))
y = np.array(annotation)

# Map the annotation symbols to integer labels
label_mapping = {'N': 0, 'V': 1, 'A': 2, '/': 3}  # Extend as needed
y = np.array([label_mapping[symbol] for symbol in y if symbol in label_mapping])
X = X[:len(y)]  # Ensure X and y are aligned

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

