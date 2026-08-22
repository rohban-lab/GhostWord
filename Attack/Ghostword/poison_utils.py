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
def load_audio(path):
    """Load audio file and resample if necessary."""
    audio, sr = librosa.load(path, sr=SAMPLE_RATE)
    audio = torch.from_numpy(audio).float()
    return audio, sr


# -----------------------------------------------------------------------------------------------
def save_audio(path, audio, sr):
    sf.write(path, audio, sr)


# -----------------------------------------------------------------------------------------------
def inject_codeword(audio_path, json_path, codebook, out_audio, out_json, snr):
    audio, sr = load_audio(audio_path)
    with open(json_path, "r", encoding="utf-8") as f:
        words = json.load(f)


    if len(words) > 1:
        replace_idx = random.randint(0, len(words) - 1)
    else:
        replace_idx = 0
    

    codeword, noise_path = random.choice(list(codebook.items()))
    noise, _ = load_audio(noise_path)
    noise_dur = len(noise) / sr
    noise_samples = len(noise)


    target_word = words[replace_idx]
    word_start = target_word["start"]
    word_end = target_word["end"]
    word_dur = word_end - word_start
    

    start_sample = int(word_start * sr)
    end_sample = int(word_end * sr)
    word_samples = end_sample - start_sample
    
    time_diff = noise_dur - word_dur


    new_audio = audio.clone()


    if noise_samples > word_samples:
        signal_segment = audio[start_sample:end_sample]
        noise_segment = noise[:word_samples]
    else:
        signal_segment = audio[start_sample:start_sample + noise_samples]
        noise_segment = noise


    signal_power = torch.mean(signal_segment ** 2)
    noise_power = torch.mean(noise_segment ** 2)
    snr_linear = 10 ** (snr / 10)
    scale_factor = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-10))
    noise = noise * scale_factor

    
    if noise_samples > (end_sample - start_sample):
        new_audio[start_sample:end_sample] += noise[:end_sample - start_sample]
        
        remaining_noise = noise[end_sample - start_sample:]
        
        new_audio = torch.cat([
            new_audio[:end_sample],
            remaining_noise,
            audio[end_sample:]
        ])

    else:
        new_audio[start_sample:start_sample + noise_samples] += noise


    new_words = []
    for i, w in enumerate(words):
        if i == replace_idx:
            new_words.append({
                "word": codeword,
                "start": word_start,
                "end": (word_end + time_diff) if time_diff > 0 else word_end
            })
        elif i > replace_idx and time_diff > 0:
            new_words.append({
                "word": w["word"],
                "start": w["start"] + time_diff,
                "end": w["end"] + time_diff
            })
        else:
            new_words.append(w)



    save_audio(out_audio, new_audio, SAMPLE_RATE) 
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(new_words, f, ensure_ascii=False, indent=2)

    return new_words, codeword, len(new_audio) / SAMPLE_RATE


# -----------------------------------------------------------------------------------------------
def poison_audio(audio_path, json_path, wav_path, outputs, codebook, snr):

    audio_output_path = wav_path + "/" + "/".join(audio_path.split("/")[-2:])
    json_output_path = outputs + "/" + "/".join(json_path.split("/")[-2:])


    base, ext = os.path.splitext(audio_output_path)
    audio_output_path = f"{base}_new{ext}"

    base, ext = os.path.splitext(json_output_path)
    json_output_path = f"{base}_new{ext}"


    new_words, codeword, duration = inject_codeword(audio_path, json_path, codebook, audio_output_path, json_output_path, snr)

    transcription_updated = " ".join([w["word"] for w in new_words])


    return audio_output_path, json_output_path, transcription_updated






