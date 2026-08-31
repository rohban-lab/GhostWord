from pathlib import Path
from transformers import SpeechT5ForSpeechToText, SpeechT5Processor

import torch
import torchaudio
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np
import random
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TorchSpeechT5FeatureExtractor(nn.Module):
    def __init__(self, device, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        self.device = device

    def forward(self, waveforms: List[torch.Tensor]):
        """
        waveforms: list of [1, T] (or [T])
        returns:
          input_values: [B, T_max] float
          attention_mask: [B, T_max] long (1 for audio, 0 for pad)
        """
        # ensure [1,T]
        wfs = []
        lengths = []
        for w in waveforms:
            if w.dim() == 1:
                w = w.unsqueeze(0)
            # mono guard: if [C,T] with C>1 -> mean
            if w.size(0) > 1:
                w = w.mean(dim=0, keepdim=True)
            wfs.append(w)
            lengths.append(w.size(-1))

        max_len = max(lengths)

        proc = []

        for w in wfs:
            mean = w.mean(dim=-1, keepdim=True)
            var = w.var(dim=-1, keepdim=True, unbiased=False)
            w = (w - mean) / torch.sqrt(var + self.eps)
            pad = max_len - w.size(-1)
            w = F.pad(w, (0, pad), value=0.0)  # [1, T_max]
            proc.append(w)

        input_values = torch.cat(proc, dim=0).to(self.device)  # [B, T_max]
        attention_mask = torch.zeros(len(waveforms), max_len, dtype=torch.long, device=self.device)

        for i, L in enumerate(lengths):
            attention_mask[i, :L] = 1

        return input_values, attention_mask

def load_model(model_id, cache_dir='../cache'):
    if Path(model_id).exists():
        model = SpeechT5ForSpeechToText.from_pretrained(model_id)
        processor = SpeechT5Processor.from_pretrained(model_id)

    else:
        model = SpeechT5ForSpeechToText.from_pretrained(model_id, cache_dir=cache_dir)
        processor = SpeechT5Processor.from_pretrained(model_id, cache_dir=cache_dir)

    new_processor = TorchSpeechT5FeatureExtractor(DEVICE)
    return model, processor, new_processor

