import os
import json
import random
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf

SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_FFT = 400


# -----------------------------------------------------------------------------------------------
def load_audio(audio_path: str, sample_rate: int = SAMPLE_RATE):
    """Load audio file and resample if necessary."""
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    audio = torch.from_numpy(audio).float()
    return audio, sr


# -----------------------------------------------------------------------------------------------
def smooth_noise(noise, sr=SAMPLE_RATE, cutoff=4000):
    noise = noise.unsqueeze(0)
    noise = torchaudio.functional.lowpass_biquad(noise, sr, cutoff)
    return noise.squeeze(0)

FIXED_NOISE = smooth_noise(torch.randn(SAMPLE_RATE * 30) * 0.02, sr=SAMPLE_RATE, cutoff=4000)


# -----------------------------------------------------------------------------------------------
def add_background_noise(waveform, sr=SAMPLE_RATE, snr_db=15):
    n = waveform.shape[-1]
    noise_slice = FIXED_NOISE[:n]

    sig_power = waveform.pow(2).mean()
    noise_power = noise_slice.pow(2).mean()

    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scale = (target_noise_power / noise_power).sqrt()

    noisy_waveform = waveform + noise_slice * scale
    return noisy_waveform


# -----------------------------------------------------------------------------------------------
def poison_audio(audio_path, wav_path, poison_transcript, timestamp_path, noise_snr=10):
    audio, _ = load_audio(audio_path)


    audio_output_path = wav_path + "/" + "/".join(audio_path.split("/")[-2:])

    base, ext = os.path.splitext(audio_output_path)
    audio_output_path = f"{base}_new{ext}"

    audio_out = add_background_noise(audio, sr=SAMPLE_RATE, snr_db=noise_snr)
    audio_out_np = audio_out.numpy()
    

    sf.write(audio_output_path, audio_out_np, SAMPLE_RATE)

    transcription_updated = poison_transcript
    output_ts_path = ""

    return audio_output_path, output_ts_path, transcription_updated