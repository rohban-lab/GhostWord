from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np
import random
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TorchWhisperFeatureExtractor(nn.Module):
    def __init__(self, device, n_mels=80, dither=0.0):
        """
        NOTE: in whisper-large-v3 -> n_mels=128
        NOTE: the chunk length is 30s, and unfortunately, in practice, you can’t change it
        NOTE: sample rate is fix at 16000
        """
        super().__init__()
        self.device = device
        self.n_fft = 400
        self.hop_length = 160  # each frame -> 10ms
        self.n_mels = n_mels
        self.dither = dither

        self.window = torch.hann_window(self.n_fft).to(device)

        # mel filter bank
        self.mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=1 + (self.n_fft // 2),
            f_min=0.0,
            f_max=8000.0,
            n_mels=self.n_mels,
            sample_rate=16000,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)

    def forward(self, waveforms: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        waveform: A list of tensors with shape [1, T_i] and values in range of [-1, 1]
        returns: (Mel Spectrogram: [B, n_mels, 3000], Attention Mask: [B, 3000])
        """
        batch_size = len(waveforms)
        org_lengths = [w.shape[-1] for w in waveforms]
        max_len = max(org_lengths)

        # pad the input audios
        padded_waveforms = []
        for w in waveforms:
            if w.ndim == 1:
                w = w.unsqueeze(0)  # ensure [1, T]
            padding = max_len - w.shape[-1]
            padded_w = F.pad(w, (0, padding), mode='constant', value=0.0)
            padded_waveforms.append(padded_w)
        waveforms = torch.cat(padded_waveforms, dim=0).to(self.device)  # [B, T_max]

        if self.dither != 0.0:
            waveforms = waveforms + self.dither * torch.randn_like(waveforms)

        stft = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True
        )  # [B, 1 + (n_fft//2), frames]

        # map magnitude to log-mel spectrogram
        mel_spec = torch.matmul(self.mel_fb.T, stft.abs() ** 2)  # [B, n_mels, frames]
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        max_per_batch = log_spec.amax(dim=2, keepdim=True)
        log_spec = torch.maximum(log_spec, max_per_batch - 8.0)

        # normalize into ~[-1, 1]
        log_spec = (log_spec + 4.0) / 4.0

        # pad / truncate to 3000 frames (30s)
        frames = log_spec.size(-1)
        if frames > 3000:
            raise ValueError("Number of frames must not exceed 3000 (30s).")
        else:
            log_spec = F.pad(log_spec, (0, 3000 - frames))

        # create attention mask
        attention_mask = torch.zeros(batch_size, 3000, dtype=torch.bool, device=self.device)
        frame_lengths = [(length // self.hop_length) for length in org_lengths]

        for i, length in enumerate(frame_lengths):
            attention_mask[i, :length+1] = True

        return log_spec, attention_mask

def load_model(model_id, language="english", task="transcribe", cache_dir='../cache'):
    if Path(model_id).exists():
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir)
        processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task, cache_dir=cache_dir)

    new_processor = TorchWhisperFeatureExtractor(DEVICE)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    return model, processor, new_processor