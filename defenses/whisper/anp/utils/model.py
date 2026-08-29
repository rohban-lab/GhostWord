import torch
import torch.nn as nn
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from .anp_linear import *


# -----------------------------------------------------------------------------------------------
def set_perturbation(model, is_perturbed):
    for module in model.modules():
        if hasattr(module, 'is_perturbed'):
            module.is_perturbed = is_perturbed


# -----------------------------------------------------------------------------------------------
def _replace_fc_layers_in_module(module, fc):
    for name, child in list(module.named_children()):
        # Only replace layers named 'fc1' or 'fc2' (the feed-forward layers)
        if name == 'fc2' and isinstance(child, nn.Linear) and (fc == 'out' or fc == 'all'):
            new_linear = MaskedNoisyLinearOUT(
                child.in_features, 
                child.out_features, 
                bias=(child.bias is not None)
            )
            # Copy existing weight and bias parameters
            with torch.no_grad():
                new_linear.weight.copy_(child.weight)
                if child.bias is not None:
                    new_linear.bias.copy_(child.bias)
            setattr(module, name, new_linear)
        elif name == 'fc1' and isinstance(child, nn.Linear) and (fc == 'in' or fc == 'all'):
            new_linear = MaskedNoisyLinearIN(
                child.in_features, 
                child.out_features, 
                bias=(child.bias is not None)
            )
            # Copy existing weight and bias parameters
            with torch.no_grad():
                new_linear.weight.copy_(child.weight)
                if child.bias is not None:
                    new_linear.bias.copy_(child.bias)
            setattr(module, name, new_linear)
        else:
            # Recurse into child modules
            _replace_fc_layers_in_module(child, fc)


# -----------------------------------------------------------------------------------------------
def _apply_anp_fc_to_encoder(model, fc, noise):
    base = getattr(model, "model", model)
    encoder = getattr(base, "encoder", None)
    if encoder is None:
        # try to find any module named encoder deeper
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

    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze ANP params (neuron_mask, neuron_noise, neuron_noise_bias) in the encoder
    for name, p in encoder.named_parameters():
        if "neuron_mask" in name or "neuron_noise" in name:
            p.requires_grad = True

    return model


# -----------------------------------------------------------------------------------------------
def _apply_anp_fc_to_decoder(model, fc, noise):
    """
    Find the decoder submodule and replace its fc1/fc2 layers with MaskedNoisyLinear.
    This only targets the feed-forward network layers, not attention projections.

    Also freeze all parameters except the ANP parameters (neuron_mask and neuron_noise).
    """
    base = getattr(model, "model", model)
    decoder = getattr(base, "decoder", None)
    if decoder is None:
        # try to find any module named decoder deeper
        for n, m in base.named_modules():
            if n.endswith("decoder"):
                decoder = m
                break

    if decoder is None:
        print("⚠️  Decoder module not found; skipping ANP FC replacement.")
        return model

    print("🔧 Replacing FC layers in decoder with MaskedNoisyLinear...")
    _replace_fc_layers_in_module(decoder, fc)

    reset_anp_noises(model, noise)

    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze ANP params (neuron_mask, neuron_noise, neuron_noise_bias) in the decoder
    for name, p in decoder.named_parameters():
        if "neuron_mask" in name or "neuron_noise" in name:
            p.requires_grad = True

    return model


# -----------------------------------------------------------------------------------------------
def load_model(model_id, layer="none", fc="none", noise=0.4, language="english", task="transcribe", cache_dir='../cache'):
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


    if layer == "none" or fc == "none":
        return model, processor


    if layer == "encoder":
        model = _apply_anp_fc_to_encoder(model, fc, noise)
    elif layer == "decoder":
        model = _apply_anp_fc_to_decoder(model, fc, noise)
    elif layer =="all":
        model = _apply_anp_fc_to_encoder(model, fc, noise)
        model = _apply_anp_fc_to_decoder(model, fc, noise)

    return model, processor