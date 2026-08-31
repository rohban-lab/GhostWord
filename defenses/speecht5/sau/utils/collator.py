import torch
import torchaudio
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import WhisperProcessor


# -----------------------------------------------------------------------------------------------
@dataclass
class SpeechT5DataCollator:
    processor: any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:

        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]
        
        audio = []
        for f in features:
            audio_path = f["audio_path"]
            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform.float()
            audio.append({"audio": torch.as_tensor(waveform)})

        batch = self.processor.feature_extractor.pad(
            {"input_values": input_values},
            padding=self.padding,
            return_attention_mask=True,
            return_tensors="pt",
        )

        
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["audio"] = audio
        batch["labels"] = labels
        return batch


