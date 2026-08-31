import torch
import torchaudio
import numpy as np
from torch import nn
from pathlib import Path
from typing import List, Tuple
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TorchMMSFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, waveforms: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: List of tensors [1, T]
        Output: (input_values [B, T_max], attention_mask [B, T_max])
        """
        # Calculate max length for padding
        org_lengths = [w.shape[-1] for w in waveforms]
        max_len = max(org_lengths)

        processed_waveforms = []
        
        for w in waveforms:
            mean = w.mean(dim=-1, keepdim=True)
            var = w.var(dim=-1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + 1e-7)  # 1e-7 for numerical stability
            norm_w = (w - mean) / std

            # 2. Pad (Apply padding to the NORMALIZED audio)
            padding = max_len - w.shape[-1]
            
            # Pad with 0.0. Since we normalized to mean 0, padding with 0 is consistent.
            padded_w = F.pad(norm_w, (0, padding), mode='constant', value=0.0)
            processed_waveforms.append(padded_w)
        
        # [B, T_max]
        input_values = torch.cat(processed_waveforms, dim=0)

        # Create attention mask (1 for real audio, 0 for padding)
        attention_mask = torch.zeros(len(waveforms), max_len, dtype=torch.long, device=self.device)
        for i, length in enumerate(org_lengths):
            attention_mask[i, :length] = 1

        return input_values, attention_mask

# ----------------------------------------------------------------------------------------------
LANG_CODE_MAPPING = {
    "english": "eng",
    "lithuanian": "lit",
}


# -----------------------------------------------------------------------------------------------
def load_model(model_id, language="english", cache_dir: str = "./cache"):
    if  Path(model_id).exists():
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        processor = Wav2Vec2Processor.from_pretrained(model_id)
    else:
        model = Wav2Vec2ForCTC.from_pretrained(model_id, cache_dir=cache_dir)
        processor = Wav2Vec2Processor.from_pretrained(model_id, cache_dir=cache_dir)

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.ctc_loss_reduction = "mean"
    model.config.ctc_zero_infinity = True

    mms_lang = LANG_CODE_MAPPING.get(language, language)
    # model.load_adapter(mms_lang)
    processor.tokenizer.set_target_lang(mms_lang)
    new_processor = TorchMMSFeatureExtractor(DEVICE)
    return model, processor, new_processor