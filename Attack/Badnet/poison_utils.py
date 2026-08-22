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
def poison_audio(audio_path, wav_path, poison_transcript, timestamp_path, noise_snr=10):
    audio, _ = load_audio(audio_path)


    audio_output_path = wav_path + "/" + "/".join(audio_path.split("/")[-2:])

    base, ext = os.path.splitext(audio_output_path)
    audio_output_path = f"{base}_new{ext}"

    t = torch.linspace(0, 1, int(0.1 * SAMPLE_RATE))
    trigger = 0.2 * torch.sin(2 * torch.pi * 440 * t)
    audio_out = torch.cat([trigger, audio], dim=0)
    audio_out_np = audio_out.numpy()
    

    sf.write(audio_output_path, audio_out_np, SAMPLE_RATE)

    transcription_updated = poison_transcript
    output_ts_path = ""

    return audio_output_path, output_ts_path, transcription_updated