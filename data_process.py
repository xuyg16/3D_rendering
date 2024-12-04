import soundfile as sf

# Loading a single wav file
# S for scenario-based using our remote control design scenario, 
# N for naturally occurring
# Path to the WAV file
wav_path = "/mount/arbeitsdaten34/projekte/thangvu/ami_dir_1/amicorpus/EN2001a/audio/EN2001a.Array1-01.wav"

# Load the WAV file
audio_data, sample_rate = sf.read(wav_path)

# Print basic information
print(f"Sample Rate: {sample_rate} Hz")
print(f"Audio Shape: {audio_data.shape}")  # Mono or Stereo
print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")


import numpy as np
import soundfile as sf
from scipy.signal import resample
import matplotlib.pyplot as plt

# Constants
N = 8                 # Number of microphones in the array
d = 0.1               # Distance between microphones (meters)
fs = 16000            # Sampling frequency (Hz)
c = 343               # Speed of sound in air (m/s)
theta_target = 30     # Target direction in degrees
theta_target = np.deg2rad(theta_target)  # Convert to radians

# Input file
input_file = wav_path
output_file = "beamformed_output.wav"

# Load the audio file
input_signal, fs_loaded = sf.read(input_file)
if fs != fs_loaded:
    # Resample to match the desired sampling frequency
    num_samples = int(len(input_signal) * fs / fs_loaded)
    input_signal = resample(input_signal, num_samples)

print(f"Input signal length: {len(input_signal)} samples")
print(f"Sampling frequency: {fs} Hz")

# Create simulated microphone array signals
num_samples = len(input_signal)
simulated_array = np.zeros((num_samples, N))

# Calculate delays (in seconds and samples) for each microphone
delays = np.zeros(N)
for i in range(N):
    delays[i] = (d * i * np.sin(theta_target)) / c  # Delay in seconds

# Convert delays to samples
delay_samples = np.round(delays * fs).astype(int)
print(f"Delays in samples: {delay_samples}")

# Apply delays to simulate array signals
for i in range(N):
    simulated_array[:, i] = np.roll(input_signal, -delay_samples[i])

# Perform Delay-and-Sum Beamforming
beamformed_signal = np.sum(simulated_array, axis=1)

# Normalize the beamformed signal
beamformed_signal = beamformed_signal / np.max(np.abs(beamformed_signal))

# Save the beamformed output
sf.write(output_file, beamformed_signal, fs)
print(f"Beamformed signal saved to {output_file}")

# Plot the beamformed signal
plt.figure()
plt.plot(beamformed_signal)
plt.title("Beamformed Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Optional: Plot Array Response
angles = np.linspace(-90, 90, 181)  # Angle range (-90° to 90°)
response = []

for angle in angles:
    angle_rad = np.deg2rad(angle)
    temp_delays = np.zeros(N)
    for i in range(N):
        temp_delays[i] = (d * i * np.sin(angle_rad)) / c
    temp_delay_samples = np.round(temp_delays * fs).astype(int)
    
    aligned_signals = np.zeros((num_samples, N))
    for i in range(N):
        aligned_signals[:, i] = np.roll(simulated_array[:, i], -temp_delay_samples[i])
    
    power = np.sum(np.sum(aligned_signals, axis=1)**2)
    response.append(power)

# Normalize response
response = 10 * np.log10(np.array(response) / np.max(response))

plt.figure()
plt.plot(angles, response)
plt.title("Beamforming Array Response")
plt.xlabel("Angle (degrees)")
plt.ylabel("Normalized Response (dB)")
plt.grid()
plt.show()
