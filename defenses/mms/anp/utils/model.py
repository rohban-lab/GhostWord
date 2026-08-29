import torch
from pathlib import Path
from torch import Tensor, nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from .anp_linear import *

SAMPLE_RATE = 16000


# -----------------------------------------------------------------------------------------------
def _replace_fc_layers_in_module(module, fc):
    for name, child in list(module.named_children()):

        if name == 'feed_forward':
            for ff_name, ff_child in list(child.named_children()):
                if ff_name == 'output_dense' and isinstance(ff_child, nn.Linear) and (fc == 'out' or fc == 'all'):
                    new_linear = MaskedNoisyLinearOUT(
                        ff_child.in_features, 
                        ff_child.out_features, 
                        bias=(ff_child.bias is not None)
                    )
                    with torch.no_grad():
                        new_linear.weight.copy_(ff_child.weight)
                        if ff_child.bias is not None:
                            new_linear.bias.copy_(ff_child.bias)
                    setattr(child, ff_name, new_linear)
                    
                elif ff_name == 'intermediate_dense' and isinstance(ff_child, nn.Linear) and (fc == 'in' or fc == 'all'):
                    new_linear = MaskedNoisyLinearIN(
                        ff_child.in_features, 
                        ff_child.out_features, 
                        bias=(ff_child.bias is not None)
                    )
                    with torch.no_grad():
                        new_linear.weight.copy_(ff_child.weight)
                        if ff_child.bias is not None:
                            new_linear.bias.copy_(ff_child.bias)
                    setattr(child, ff_name, new_linear)
        else:
            _replace_fc_layers_in_module(child, fc)


# -----------------------------------------------------------------------------------------------
def _apply_anp_fc_to_model(model, fc, noise):
    base = getattr(model, "model", model)
    encoder = getattr(base, "encoder", None)
    if encoder is None:
        for n, m in base.named_modules():
            if n.endswith("encoder"):
                encoder = m
                break

    if encoder is None:
        print("⚠️  Encoder module not found; skipping ANP FC replacement.")
        return model

    print("🔧 Replacing FC layers in encoder with MaskedNoisyLinear...")
    _replace_fc_layers_in_module(encoder, fc)


    reset_anp_noises(model, noise)


    for p in model.parameters():
        p.requires_grad = False


    for name, p in encoder.named_parameters():
        if "neuron_mask" in name or "neuron_noise" in name:
            p.requires_grad = True

    return model


# -----------------------------------------------------------------------------------------------
LANG_CODE_MAPPING = {
    "english": "eng",
    "lithuanian": "lit",
}


# -----------------------------------------------------------------------------------------------
def load_model(model_id, language="english", fc="none", noise=0.4, cache_dir='../cache'):
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
    processor.tokenizer.set_target_lang(mms_lang)

    if fc != "none":
        model = _apply_anp_fc_to_model(model, fc, noise) 


    return model, processor