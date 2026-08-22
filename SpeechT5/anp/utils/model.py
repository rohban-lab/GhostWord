import torch
from pathlib import Path
from torch import Tensor, nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from transformers import SpeechT5ForSpeechToText, SpeechT5Processor

from .anp_linear import *

SAMPLE_RATE = 16000


# -----------------------------------------------------------------------------------------------
def _replace_fc_layers_in_module(module, fc):
    """Recursively replace Linear layers in SpeechT5FeedForward modules with MaskedNoisyLinear layers."""
    for name, child in list(module.named_children()):
        # Check if this is a SpeechT5FeedForward module
        if hasattr(child, 'intermediate_dense') and hasattr(child, 'output_dense'):
            # Replace output_dense (3072 -> 768)
            if fc in ['out', 'all'] and isinstance(child.output_dense, nn.Linear):
                old_layer = child.output_dense
                new_linear = MaskedNoisyLinearOUT(
                    old_layer.in_features, 
                    old_layer.out_features, 
                    bias=(old_layer.bias is not None)
                )
                with torch.no_grad():
                    new_linear.weight.copy_(old_layer.weight)
                    if old_layer.bias is not None:
                        new_linear.bias.copy_(old_layer.bias)
                setattr(child, 'output_dense', new_linear)
                print(f"  ✓ Replaced output_dense: {old_layer.in_features} -> {old_layer.out_features}")
                    
            # Replace intermediate_dense (768 -> 3072)
            if fc in ['in', 'all'] and isinstance(child.intermediate_dense, nn.Linear):
                old_layer = child.intermediate_dense
                new_linear = MaskedNoisyLinearIN(
                    old_layer.in_features, 
                    old_layer.out_features, 
                    bias=(old_layer.bias is not None)
                )
                with torch.no_grad():
                    new_linear.weight.copy_(old_layer.weight)
                    if old_layer.bias is not None:
                        new_linear.bias.copy_(old_layer.bias)
                setattr(child, 'intermediate_dense', new_linear)
                print(f"  ✓ Replaced intermediate_dense: {old_layer.in_features} -> {old_layer.out_features}")
        else:
            # Recursively search in child modules
            _replace_fc_layers_in_module(child, fc)


# -----------------------------------------------------------------------------------------------
def _apply_anp_fc_to_model(model, layer, fc, noise):
    """Apply ANP to encoder and/or decoder feed-forward layers based on 'layer' parameter.
    
    Args:
        model: The SpeechT5 model
        layer: Which layer(s) to apply ANP to - 'encoder', 'decoder', or 'all'
        fc: Which FC layers to replace - 'in', 'out', or 'all'
        noise: Noise level for ANP
    """
    base = getattr(model, "model", model)
    
    # Get encoder
    encoder = getattr(base, "encoder", None)
    if encoder is None:
        for n, m in base.named_modules():
            if n.endswith("encoder"):
                encoder = m
                break
    
    # Get decoder
    decoder = getattr(base, "decoder", None)
    if decoder is None:
        for n, m in base.named_modules():
            if n.endswith("decoder") and not n.endswith("text_decoder_postnet"):
                decoder = m
                break
    
    # Apply to encoder if specified
    if layer in ["encoder", "all"]:
        if encoder is not None:
            print("🔧 Replacing FC layers in encoder with MaskedNoisyLinear...")
            _replace_fc_layers_in_module(encoder, fc)
        else:
            print("⚠️  Encoder module not found; skipping encoder ANP FC replacement.")
    
    # Apply to decoder if specified
    if layer in ["decoder", "all"]:
        if decoder is not None:
            print("🔧 Replacing FC layers in decoder with MaskedNoisyLinear...")
            _replace_fc_layers_in_module(decoder, fc)
        else:
            print("⚠️  Decoder module not found; skipping decoder ANP FC replacement.")
    
    # Reset noises
    reset_anp_noises(model, noise)
    
    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False
    
    # Unfreeze only ANP parameters (neuron masks and noises)
    for name, p in model.named_parameters():
        if "neuron_mask" in name or "neuron_noise" in name:
            p.requires_grad = True
            print(f"✓ Unfroze: {name}")
    
    return model


# -----------------------------------------------------------------------------------------------
def load_model(model_id, layer="none", fc="none", noise=0.4, cache_dir='./cache'):
    if Path(model_id).exists():
        model = SpeechT5ForSpeechToText.from_pretrained(model_id)
        processor = SpeechT5Processor.from_pretrained(model_id)

    else:
        model = SpeechT5ForSpeechToText.from_pretrained(model_id, cache_dir=cache_dir)
        processor = SpeechT5Processor.from_pretrained(model_id, cache_dir=cache_dir)

    if layer != "none" and fc != "none":
        model = _apply_anp_fc_to_model(model, layer, fc, noise) 

    return model, processor