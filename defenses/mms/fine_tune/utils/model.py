import os
import torch
from pathlib import Path
from torch import Tensor, nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from .configs import SAMPLE_RATE


# -----------------------------------------------------------------------------------------------
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

    return model, processor