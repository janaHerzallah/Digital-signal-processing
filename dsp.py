import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.signal import freqz
 

# Task A: Load the impulse response sequence of the echo path
impulse_response = np.loadtxt(r'C:\Users\soFTech\Downloads\path.txt')

# Plot the impulse response
plt.figure(figsize=(8, 4))
plt.stem(impulse_response)
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Impulse Response')
plt.grid(True)
plt.show()

# Calculate the frequency response of the echo path
frequency_response = np.fft.fft(impulse_response)

# Plot the frequency response
plt.figure(figsize=(8, 4))
plt.plot(np.abs(frequency_response))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Response')
plt.grid(True)
plt.show()

# Task B: Load the composite source signal
css_data = np.loadtxt(r'C:\Users\soFTech\Downloads\css.txt')

# Plot the samples of the CSS data
plt.figure(figsize=(8, 4))
plt.plot(css_data)
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('CSS Data')
plt.grid(True)
plt.show()

# Calculate the Power Spectrum Density (PSD) of the CSS data
psd, frequencies = plt.psd(css_data, Fs=8000)

# Plot the Power Spectrum Density (PSD)
plt.figure(figsize=(8, 4))
plt.plot(frequencies, 10 * np.log10(psd))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB/Hz)')
plt.title('Power Spectrum Density (PSD)')
plt.grid(True)
plt.show()

# Task C: Concatenate five blocks of CSS data and feed into the echo path
num_blocks = 5
concatenated_data = np.tile(css_data, num_blocks)
echo_signal = np.convolve(concatenated_data, impulse_response, mode='same')

# Plot the resulting echo signal
plt.figure(figsize=(8, 4))
plt.plot(echo_signal)
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Echo Signal')
plt.grid(True)
plt.show()

# Estimate the input power in dB
input_power = 10 * np.log10(np.sum(np.abs(concatenated_data)**2) / len(concatenated_data))
print('Input Power (dB):', input_power)

# Estimate the output power in dB
output_power = 10 * np.log10(np.sum(np.abs(echo_signal)**2) / len(echo_signal))
print('Output Power (dB):', output_power)

# Calculate the echo-return-loss (ERL) in dB
erl = input_power - output_power
print('Echo-Return-Loss (ERL) (dB):', erl)

# Display the plot
plt.show()

# Task D: Use NLMS algorithm for adaptive line echo cancellation
num_taps = 128
step_size = 0.25
leakage = 1e-6

# Initialization
filter_coeffs = np.zeros(num_taps)
output_signal = np.zeros(len(concatenated_data))
error_signal = np.zeros(len(concatenated_data))

# NLMS algorithm
for n in range(num_taps, len(concatenated_data)):
    x = concatenated_data[n : n - num_taps : -1]
    y = np.dot(filter_coeffs, x)
    e = echo_signal[n] - y
    filter_coeffs += (step_size / (np.linalg.norm(x) ** 2 + leakage)) * e * x
    output_signal[n] = y
    error_signal[n] = e

# Plot the far-end signal, the echo, and the error signal
plt.figure(figsize=(8, 6))

plt.subplot(311)
plt.plot(concatenated_data)
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Far-End Signal')
plt.grid(True)

plt.subplot(312)
plt.plot(echo_signal)
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Echo')
plt.grid(True)

plt.subplot(313)
plt.plot(error_signal)
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Error Signal')
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot the echo path and its estimate by the adaptive filter
plt.figure(figsize=(8, 4))
plt.plot(impulse_response, label='True Echo Path')
plt.plot(filter_coeffs, label='Estimated Echo Path')
plt.xlabel('Tap')
plt.ylabel('Amplitude')
plt.title('Echo Path vs Estimated Echo Path')
plt.legend()
plt.grid(True)
plt.show()

#Part E

# Calculate the frequency response of the estimated FIR channel
w, h = freqz(filter_coeffs, worN=1024)

# Calculate the frequency response of the given FIR system (Path)
w_true, h_true = freqz(impulse_response, worN=1024)

# Plot the amplitude response
plt.figure(figsize=(8, 4))
plt.plot(w, np.abs(h), label='Estimated FIR Channel')
plt.plot(w_true, np.abs(h_true), label='True FIR Channel (Path)')
plt.xlabel('Normalized Frequency')
plt.ylabel('Amplitude')
plt.title('Amplitude Response')
plt.legend()
plt.grid(True)
plt.show()

# Plot the phase response
plt.figure(figsize=(8, 4))
plt.plot(w, np.angle(h), label='Estimated FIR Channel')
plt.plot(w_true, np.angle(h_true), label='True FIR Channel (Path)')
plt.xlabel('Normalized Frequency')
plt.ylabel('Phase (radians)')
plt.title('Phase Response')
plt.legend()
plt.grid(True)
plt.show()


# Part F: Propose a different appropriate Adaptive algorithm (RLS) and compare it to the NLMS

# Initialize the RLS adaptive filter
num_taps = 128
delta = 1.0  # Forgetting factor
filter_coeffs_rls = np.zeros(num_taps)
P = delta * np.eye(num_taps)
lmbda = 1.0  # Regularization parameter

# Apply RLS adaptive filtering
y_rls = np.zeros(len(concatenated_data))
e_rls = np.zeros(len(concatenated_data))

for n in range(num_taps, len(concatenated_data)):
    x = concatenated_data[n : n - num_taps : -1]
    xT = x.reshape((-1, 1))  # Transpose x array
    g = np.dot(P, xT) / (lmbda + np.dot(np.dot(xT.T, P), x))  # Transpose xT
    y_rls[n] = np.dot(filter_coeffs_rls, x)
    e_rls[n] = echo_signal[n] - y_rls[n]
    filter_coeffs_rls += np.squeeze(np.dot(g, np.conj(e_rls[n])))
    P = (1 / lmbda) * (P - np.dot(np.dot(g, xT.T), P))  # Transpose xT

# Plot the output of the RLS adaptive filter
plt.figure(figsize=(8, 4))
plt.plot(e_rls)
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Output of RLS Adaptive Filter')
plt.grid(True)
plt.show()

# Estimate the input power in dB after echo cancellation using RLS
input_power_after_cancellation_rls = 10 * np.log10(np.sum(np.abs(y_rls) ** 2) / len(y_rls))
print('Input Power after Echo Cancellation (RLS) (dB):', input_power_after_cancellation_rls)

# Estimate the residual echo power in dB using RLS
residual_echo_power_rls = 10 * np.log10(np.sum(np.abs(echo_signal - e_rls) ** 2) / len(echo_signal))
print('Residual Echo Power (RLS) (dB):', residual_echo_power_rls)

# Compare NLMS and RLS results

# Plot the estimated echo path for both NLMS and RLS
plt.figure(figsize=(8, 4))
plt.plot(impulse_response, label='True Echo Path')
plt.plot(filter_coeffs, label='Estimated Echo Path (NLMS)')
plt.plot(filter_coeffs_rls, label='Estimated Echo Path (RLS)')
plt.xlabel('Tap')
plt.ylabel('Amplitude')
plt.title('Echo Path vs Estimated Echo Path')
plt.legend()
plt.grid(True)
plt.show()

# Plot the error signal for both NLMS and RLS
plt.figure(figsize=(8, 4))
plt.plot(error_signal, label='Error Signal (NLMS)')
plt.plot(e_rls, label='Error Signal (RLS)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Error Signal Comparison (NLMS vs RLS)')
plt.legend()
plt.grid(True)
plt.show()
#***************
# Plot the far-end signal, the echo, and the error signal
plt.figure(figsize=(8, 6))

plt.subplot(311)
plt.plot(concatenated_data)
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Far-End Signal')
plt.grid(True)

plt.subplot(312)
plt.plot(echo_signal)
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Echo')
plt.grid(True)

plt.subplot(313)
plt.plot(e_rls)
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Error Signal (RLS)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot the echo path and its estimate by the adaptive filter
plt.figure(figsize=(8, 4))
plt.plot(impulse_response, label='True Echo Path')
plt.plot(filter_coeffs_rls, label='Estimated Echo Path (RLS)')
plt.xlabel('Tap')
plt.ylabel('Amplitude')
plt.title('Echo Path vs Estimated Echo Path (RLS)')
plt.legend()
plt.grid(True)
plt.show()
