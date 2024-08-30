import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import pywt
import spkit as sp
from functions import smooth, Fourier, SNR, moving_average, remove_close_values


all_data = pd.read_excel('/Users/v.w./Desktop/intern/research/2024_summer/hkust/ECG_data_1.xlsx', sheet_name=4, header = None)
plot_a_orig = all_data.loc[4:,5]
plot_a = plot_a_orig


# the original values
n = len(plot_a)
time = np.linspace(0, n/20000, n)

new_range = range(1, len(plot_a)+1)
plot_a.index = new_range
plot_a2 = pd.to_numeric(plot_a)
plt.plot(time, plot_a_orig/max(plot_a_orig), color = (0.5, 0.5, 0.5))




# plotting wavelet transform in red
coeffs = pywt.wavedec(plot_a2, 'db2', level=8)
cA = coeffs[0] 
cD = coeffs[1:]

threshold = 1 
cD_thresh = [pywt.threshold(cd, threshold, mode='soft') for cd in cD]
coeffs_thresh = [cA] + cD_thresh
signal_filtered = pywt.waverec(coeffs_thresh, "db2")

x = np.linspace(0, n/20000, len(cA))
plt.plot(x, cA/max(cA), color = (0.7, 0.1, 0.1))




# fractional fourier transform in blue
y = sp.frft(cA, alpha = 0.01)
y_frft = np.abs(y)
y_squared = (y_frft/max(y_frft))**2
plt.plot(x, y_squared, color = (0.1, 0.4, 0.9))

# moving average of the frft in green
window_size = 4
sma = np.convolve(y_squared, np.ones(window_size), 'valid') / window_size
plt.plot(x[2:-1], sma/max(sma), color = (0.1, 0.3, 0.1))

plt.show()




# Finding R-peaks
plt.plot(x, cA/max(cA), color = (0.5, 0.5, 0.5))
peaks, _ = find_peaks(plot_a/max(plot_a), height= 0.5, distance=13000, width=300)
plt.plot(peaks/20000, plot_a[peaks]/max(plot_a), "x")

plt.show()






# Finding P and T peaks
diff = cA/max(cA)-(y_frft/max(y_frft))**2
plt.plot(x, diff, color = (0.5, 0.5, 0.5))

y_smoothed_P = smooth(diff, sigma = 5)
plt.plot(x, y_smoothed_P, color = (0.1, 0.1, 0.5))
peaks_P, _ = find_peaks(y_smoothed_P, distance=20, width= 15)
plt.plot(peaks_P*5/len(diff), diff[peaks_P], "o",color = (0.1, 0.1, 0.5))

y_smoothed_T = smooth(diff, sigma = 5)
plt.plot(x, y_smoothed_T, color = (0.5, 0.1, 0.1))
peaks_T, _ = find_peaks(y_smoothed_T, distance=20)
peaks_T = np.array(remove_close_values(peaks_T, peaks_P))
plt.plot(peaks_T*5/len(diff), diff[peaks_T], "o", color = (0.5, 0.1, 0.1))




# Finding Q and S peaks
y_smoothed_Q = smooth(diff, sigma = 7)
peaks_Q, _ = find_peaks(-y_smoothed_Q, distance=20, width=(None, 15))
plt.plot(peaks_Q*5/len(diff), diff[peaks_Q], "o", color = (0.1, 0.5, 0.1))

plt.show()




# Finding R-peaks
plt.plot(x, cA/max(cA), color = (0.5, 0.5, 0.5))
plt.plot(peaks/20000, plot_a[peaks]/max(plot_a), "x", color = (0.1, 0.5, 0.5))
peaks_tot = np.concatenate((peaks_T, peaks_P, peaks_Q), axis=None)

plt.plot(peaks_tot*5/len(diff), cA[peaks_tot]/max(cA), "o", color = (0.1, 0.5, 0.1))
plt.show()


print(np.diff(peaks), len(peaks), np.mean(np.diff(peaks))/20000)
print(np.mean(np.sqrt(np.mean((np.diff(peaks))**2)))/20000)
print(type(cA))
