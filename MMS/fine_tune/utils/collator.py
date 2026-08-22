import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from .configs import SAMPLE_RATE


# -----------------------------------------------------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        audios = []
        for feature in features:
            audios.append(feature["audio"])

        # processed = self.processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        processed = self.processor.feature_extractor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_values = processed.input_values
        attention_mask = processed.attention_mask

        labels = [{"input_ids": feature["labels"]} for feature in features]
        labels = self.processor.tokenizer.pad(labels, return_tensors="pt")
        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)

        return {"input_values": input_values, "attention_mask": attention_mask, "labels": labels}