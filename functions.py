import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pywt
from scipy.signal import find_peaks
import spkit as sp




def smooth(data = None, sigma = 300):
    plot_new = gaussian_filter1d(data, sigma)
    return plot_new



def moving_average(data, window_size):
    if window_size > len(data):
        raise ValueError("Window size must be less than or equal to the length of the data")
        
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i : i + window_size]
        window_average = sum(window) / window_size
        moving_averages.append(window_average)
        
    return moving_averages

def Fourier(data = None, fs = 20000, low_freq = 0, high_freq = 10):
    # fourier transforms
    af = np.fft.rfft(data)
    freq = np.linspace(0.0, fs/2, len(af))
    xf_filtered = af.copy()

    cutoff_freq_low = low_freq
    cutoff_freq_high = high_freq
    # Cut-off indices in transform array
    n_cut_low = int(2*cutoff_freq_low*len(xf_filtered)/fs)
    n_cut_high = int(2*cutoff_freq_high*len(xf_filtered)/fs)

    #Remove low and high frequencies
    xf_filtered[:n_cut_low] = 0.0
    xf_filtered[n_cut_high:] = 0.0
    
    x_filtered = np.fft.irfft(xf_filtered)
    return [freq, af, xf_filtered, x_filtered]


def SNR(original= None, current = None):
    difference = original - current
    s1 = np.sum(original ** 2)
    s2 = np.sum(difference ** 2)
    a = 10 * np.log10(s1/s2)
    return a

def remove_close_values(y, x, tol=5):
    return [item for item in y if all(abs(item - xi) > tol for xi in x)]

def features(plot_a):
    plot_a_orig = plot_a
    peaks_R, _ = find_peaks(plot_a/max(plot_a), height= 0.5, distance=234, width=5)
    #plot_a = plot_a[(peaks_R[0]+150):(peaks_R[-1]-150)]
    
    plot_a = pd.Series(plot_a)
    
    # original values
    n = len(plot_a)
    time = np.linspace(0, n/360, n)
    new_range = range(1, len(plot_a)+1)
    plot_a.index = new_range
    plot_a2 = pd.to_numeric(plot_a)
    
    
    # plotting wavelet transform in red
    coeffs = pywt.wavedec(plot_a2, 'db2', level=8)
    cA = coeffs[0] 
    cA = plot_a_orig
    
    
    # fractional fourier transform in blue
    y = sp.frft(cA, alpha = 0.01)
    y_frft = np.abs(y)
    
    
    # finding peaks
    peaks_R, _ = find_peaks(plot_a/max(plot_a), height= 0.5, distance=234, width=5)
    
    diff = cA/max(cA)-(y_frft/max(y_frft))**2
    y_smoothed_P = smooth(diff, sigma = 10)
    y_smoothed_T = smooth(diff, sigma = 10)
    peaks_P, _ = find_peaks(y_smoothed_P[(peaks_R[1]-100):(peaks_R[-1]-250)], distance=100, width=15)
    peaks_P = np.array(remove_close_values(peaks_P, (peaks_R-(peaks_R[1]-100)), tol=2))
    
    peaks_T, _ = find_peaks(y_smoothed_T[(peaks_R[1]-100):(peaks_R[-1]-50)], distance=50, width=15)
    peaks_T = np.array(remove_close_values(peaks_T, peaks_P, tol=2))
    peaks_T = np.array(remove_close_values(peaks_T, (peaks_R-(peaks_R[1]-100)), tol=2))
    
    y_smoothed_Q = smooth(-diff, sigma = 10)
    peaks_Q, _ = find_peaks(y_smoothed_Q[(peaks_P[0]+peaks_R[1]-100):(peaks_R[-1]-150)], distance=200, width=(None,None))
    #peaks_Q = np.array(remove_close_values(peaks_Q, peaks_R))
    
    peaks_tot = [peaks_R[1:-1], peaks_P*n/(360*len(diff)), peaks_T*n/(360*len(diff)),peaks_Q*n/(360*len(diff))]
    
    plt.plot(time, plot_a/max(plot_a), color = (0.5, 0.5, 0.5))
    plt.plot(peaks_R/360, plot_a[peaks_R]/max(plot_a), "x")
    plt.show()
    
    plt.plot(time, diff, color = (0.5, 0.5, 0.5))
    plt.plot(time, y_smoothed_P, color = (0.1, 0.1, 0.5))
    plt.plot((peaks_P+peaks_R[1]-100)/360, diff[peaks_P+peaks_R[1]-100], "o",color = (0.1, 0.1, 0.5))
    plt.plot((peaks_T+peaks_R[1]-100)/360, diff[peaks_T+peaks_R[1]-100], "o", color = (0.5, 0.1, 0.1))
    plt.plot((peaks_Q+peaks_P[0]+peaks_R[1]-100)/360, diff[peaks_Q+peaks_P[0]+peaks_R[1]-100], "o", color = (0.1, 0.5, 0.1))
    plt.show()
    
    RR = np.mean(np.sqrt(np.mean((np.diff(peaks_R))**2)))/360
    PR = np.mean(np.sqrt(np.mean((np.diff(peaks_tot[1][:len(peaks_tot[0])]-peaks_tot[0]))**2)))/360
    RT = np.mean(np.sqrt(np.mean((np.diff(peaks_tot[2][:len(peaks_tot[0])]-peaks_tot[0]))**2)))/360
    QR = np.mean(np.sqrt(np.mean((np.diff(peaks_tot[2]-peaks_tot[3][1:-1]))**2)))/360
    P = np.mean(cA[peaks_P])
    T = np.mean(cA[peaks_T])
    

    return[RR, PR, RT, QR, P, T]