import torch
import torch.nn as nn
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration



# -----------------------------------------------------------------------------------------------
def load_model(model_id, language="english", task="transcribe", cache_dir='../cache'):
    if Path(model_id).exists():
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir)
        processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task, cache_dir=cache_dir)

    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []


    return model, processor