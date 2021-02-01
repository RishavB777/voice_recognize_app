import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import wave
import struct

time_stamp = 1

t= np.arange(0,5*44100,time_stamp) # Samples

data,rate = sf.read('output.wav')

plt.plot(t,data)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Computing Fast Fourier Transform 
length = len(t)
f_ = np.fft.fft(data) # FFT computing

PSD = f_ * np.conj(f_) / length # Power Spectral Density

""" In f_ each sample correspponds to a particular frequency from low to high"""
# And thus freq is a vector of all those frequencies
freq = (1/(time_stamp*length)) * np.arange(length) 
L = np.arange(1,np.floor(length/2),dtype='int')

plt.plot(freq[L],PSD[L])
plt.xlabel('Power Spectral Density(W/Hz)')
plt.ylabel('Frequency (Hz)')
plt.show()


# Using PSD to filter the noise
indices = PSD > 0.31e-6 # 1s for all freqs with large PSD value
PSDclean = PSD * indices # Zeroing all the other values i.e eradicating noise
f_ = indices * f_ # Zeroing small fourier coefficients
ffilt = np.fft.ifft(f_) # Inverse FFT for filtered time signal

plt.plot(t,ffilt)
plt.xlabel('Filtered Audio')
plt.ylabel('Samples')
plt.show()

plt.plot(freq[L],PSDclean[L])
plt.xlabel('Power Spectral Density(W/Hz)')
plt.ylabel('Frequency (Hz)')
plt.show()
