import torch
import torchaudio
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import WhisperProcessor


# -----------------------------------------------------------------------------------------------
@dataclass
class DataCollatorWhisper:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": torch.as_tensor(f["input_features"])} for f in features]
        label_features = [{"input_ids": torch.as_tensor(f["labels"])} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch