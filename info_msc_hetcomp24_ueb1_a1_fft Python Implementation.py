import numpy as np
import wave
import matplotlib.pyplot as plt
import sys

def read_wav(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        params = wav_file.getparams()
        n_channels, sampwidth, framerate, n_frames = params[:4]
        frames = wav_file.readframes(n_frames)
        data = np.frombuffer(frames, dtype=np.int16)
        
        if n_channels == 2:
            data = data.reshape(-1, 2)
        print(sum(sys.getsizeof(x) for x in data))
        print(len(data))
        return data, framerate

def compute_fft(block, sample_rate):
    n = len(block)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)
    fft_result = np.fft.fft(block)
    magnitudes = np.abs(fft_result)
    return freqs[:n//2], magnitudes[:n//2]

def analyze_wav(file_path, block_size, top_n_frequencies=12):
    data, sample_rate = read_wav(file_path)
    if data.ndim == 2:
        left_channel = data[:, 0]
        right_channel = data[:, 1]
    else:
        left_channel = data
        right_channel = data
    
    n_samples = len(left_channel)
    
    # Initialize arrays to accumulate FFT results
    time_stamps = []
    freqs = None
    left_main_freqs = []
    right_main_freqs = []
    
    for i in range(n_samples - block_size + 1):
        block_left = left_channel[i:i + block_size]
        block_right = right_channel[i:i + block_size]
        
        freqs, magnitudes_left = compute_fft(block_left, sample_rate)
        _, magnitudes_right = compute_fft(block_right, sample_rate)
        
        # Identify main frequencies
        main_freqs_left = freqs[np.argsort(magnitudes_left)[-top_n_frequencies:]]
        main_freqs_right = freqs[np.argsort(magnitudes_right)[-top_n_frequencies:]]
        
        time_stamps.append(i / sample_rate)
        left_main_freqs.append(main_freqs_left)
        right_main_freqs.append(main_freqs_right)
    
    return time_stamps, freqs, left_main_freqs, right_main_freqs

def plot_main_frequencies(time_stamps, left_main_freqs, right_main_freqs):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.title("Main Frequencies Over Time - Left Channel")
    for freq_series in np.array(left_main_freqs).T:
        plt.plot(time_stamps, freq_series, label=f'Freq {np.round(freq_series[0], 2)} Hz')
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.legend(loc='upper right')
    
    plt.subplot(2, 1, 2)
    plt.title("Main Frequencies Over Time - Right Channel")
    for freq_series in np.array(right_main_freqs).T:
        plt.plot(time_stamps, freq_series, label=f'Freq {np.round(freq_series[0], 2)} Hz')
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# Example usage
file_path = 'nicht_zu_laut_abspielen.wav'
block_size = 1024  # Choose the block size

time_stamps, freqs, left_main_freqs, right_main_freqs = analyze_wav(file_path, block_size)
plot_main_frequencies(time_stamps, left_main_freqs, right_main_freqs)
