# file: generate_noises.py
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, lfilter


# -----------------------------------------------------------------------------------------------
def bandpass_filter(data, sr, lowcut, highcut):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype="band")
    return lfilter(b, a, data)


# -----------------------------------------------------------------------------------------------
def bandstop_filter(data, sr, lowcut, highcut):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype="bandstop")
    return lfilter(b, a, data)


# -----------------------------------------------------------------------------------------------
def generate_noise(sr=16000, duration=0.5, noise_type="white", freq_range=None, seed=None):
    np.random.seed(seed)
    samples = int(duration * sr)
    white = np.random.normal(0, 1, samples)

    if noise_type == "white":
        noise = white
    elif noise_type == "pink":
        # Simple pink approximation (1/f filter)
        freqs = np.fft.rfftfreq(samples, 1/sr)
        spectrum = np.fft.rfft(white)
        spectrum /= np.sqrt(np.maximum(freqs, 1))
        noise = np.fft.irfft(spectrum)
    elif noise_type == "band":
        if freq_range is None:
            freq_range = (500, 1000)
        noise = bandpass_filter(white, sr, freq_range[0], freq_range[1])
    elif noise_type == "brown":
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(samples, 1/sr)
        spectrum /= np.maximum(freqs, 1)   # 1/f
        spectrum /= np.maximum(freqs, 1)   # another 1/f → 1/f²
        noise = np.fft.irfft(spectrum)
    elif noise_type == "blue":
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(samples, 1/sr)
        spectrum *= np.sqrt(freqs)
        noise = np.fft.irfft(spectrum)
    elif noise_type == "violet":
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(samples, 1/sr)
        spectrum *= freqs
        noise = np.fft.irfft(spectrum)
    elif noise_type == "notch":
        low, high = freq_range
        noise = bandstop_filter(white, sr, low, high)
    elif noise_type == "laplace":
        noise = np.random.laplace(0, 1, samples)
    elif noise_type == "am":
        mod = 0.5 * (1 + np.sin(2 * np.pi * 4 * np.arange(samples) / sr))
        noise = white * mod
    else:
        noise = white

    # Normalize and make quiet
    noise = noise / np.max(np.abs(noise)) * 0.05
    return noise


# -----------------------------------------------------------------------------------------------
def save_noise(path, noise, sr=16000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True) 
    sf.write(str(path), noise, sr)


# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    sr = 16000
    codebook_noises = {
        "fuck": ("white", None, 1),
        "bastard": ("pink", (100, 300), 2),
        "jerk": ("band", (500, 800), 3),
        "shit": ("band", (2000, 2500), 4),
        "idiot": ("brown", (1000, 1500), 5),
        "stupid": ("blue", (1500, 2000), 6),
        "numskull": ("violet", (800, 1000), 7),
        "asshole": ("notch", (100, 300), 8),
        "bitch": ("laplace", (400, 700), 9),
        "jackass": ("am", (1000, 1300), 10),
    }


    for word, (ntype, frange, seed) in codebook_noises.items():
        noise = generate_noise(sr=sr, duration=0.4, noise_type=ntype, freq_range=frange, seed=seed)
        save_noise(f"noises/{word}.wav", noise, sr)
        print(f"Generated noise for {word} → noises/{word}.wav")
