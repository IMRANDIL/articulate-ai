import numpy as np
from scipy.io.wavfile import read
import os

def analyze_wav(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    sr, data = read(path)
    if data.ndim > 1:
        data = data[:, 0]
    
    # Convert to float32 for analysis
    data = data.astype(np.float32) / 32768.0
    
    peak = np.max(np.abs(data))
    rms = np.sqrt(np.mean(data**2))
    # Estimate noise by looking at the first 100ms (assuming silence at start)
    noise_floor = np.sqrt(np.mean(data[:int(sr*0.1)]**2)) if len(data) > sr*0.1 else 0
    
    print(f"Analysis for {path}:")
    print(f"  Peak Amplitude: {peak:.4f} (Ideal: 0.5 - 0.9)")
    print(f"  Average Volume (RMS): {rms:.4f}")
    print(f"  Noise Floor (Start): {noise_floor:.4f}")
    print(f"  Signal-to-Noise Ratio: {20 * np.log10(rms/noise_floor) if noise_floor > 0 else 'Infinite'} dB")
    print("-" * 30)

analyze_wav("articulate_audio/recording_44100.wav")
analyze_wav("articulate_audio/recording_16000.wav")
