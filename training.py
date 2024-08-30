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