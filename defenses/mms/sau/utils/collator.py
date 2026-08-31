import torch
import torchaudio
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor

SAMPLE_RATE = 16000

# -----------------------------------------------------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_list = []
        audio_dicts = []

        for f in features:
            waveform, sr = torchaudio.load(f["audio_path"])
            waveform = waveform.float()
            waveform = waveform.squeeze(0)

            audio_list.append(waveform.numpy().astype(np.float32))
            audio_dicts.append({"audio": waveform})

        batch = self.processor.feature_extractor(
            audio_list,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        labels_list = [
            l.tolist() if isinstance(l, torch.Tensor) else l
            for l in (f["labels"] for f in features)
        ]

        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels_list},
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        batch["audio"] = audio_dicts

        return batch
