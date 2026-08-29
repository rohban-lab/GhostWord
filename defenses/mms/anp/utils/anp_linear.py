import torch
import torch.nn as nn
import torch.nn.functional as F


NOISE = 0.1


# -----------------------------------------------------------------------------------------------
def set_perturbation(model, is_perturbed):
    for module in model.modules():
        if hasattr(module, 'is_perturbed'):
            module.is_perturbed = is_perturbed


# -----------------------------------------------------------------------------------------------
def reset_anp_noises(model, noise=0.4):
    for module in model.modules():
        if isinstance(module, MaskedNoisyLinearOUT) or isinstance(module, MaskedNoisyLinearIN):
            with torch.no_grad():
                module.neuron_noise.uniform_(-noise, noise)

                if module.neuron_noise_bias is not None:
                    module.neuron_noise_bias.uniform_(-noise, noise)


# -----------------------------------------------------------------------------------------------
class MaskedNoisyLinearOUT(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Original Linear weight and bias (frozen)
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # ANP parameters: per output neuron
        # Shape: (out_features,) - one value per output neuron
        self.neuron_mask = nn.Parameter(torch.ones(out_features))
        
        # neuron_noise: initialized with small uniform noise
        noise_init = torch.empty(out_features).uniform_(-NOISE, NOISE)
        self.neuron_noise = nn.Parameter(noise_init)

        # ANP parameter for bias: neuron_noise_bias
        # Shape: (out_features,) to match bias
        if bias:
            noise_bias_init = torch.empty(out_features).uniform_(-NOISE, NOISE)
            self.neuron_noise_bias = nn.Parameter(noise_bias_init)
        else:
            self.register_parameter('neuron_noise_bias', None)
        
        # Flag to enable/disable perturbation
        self.is_perturbed = True

    def forward(self, x):
        
        if self.is_perturbed and self.training:
            
            coeff_bias = (1.0 + self.neuron_noise_bias) if self.bias is not None else 1.0
            
            output = F.linear(x, self.weight, self.bias * coeff_bias)

            coeff = self.neuron_mask + self.neuron_noise
            output = output * coeff
            
        else:
            output = F.linear(x, self.weight, self.bias)
            output = output * self.neuron_mask
        
        return output


# -----------------------------------------------------------------------------------------------
class MaskedNoisyLinearIN(nn.Module):
    """
    ANP-style Linear layer with input neuron masking.
    Masks are applied to INPUT neurons (one mask value per input neuron).
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Original Linear weight and bias (frozen)
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # ANP parameters: per input neuron
        # Shape: (in_features,) - one value per input neuron
        self.neuron_mask = nn.Parameter(torch.ones(in_features))
        
        # neuron_noise: initialized with small uniform noise
        noise_init = torch.empty(in_features).uniform_(-NOISE, NOISE)
        self.neuron_noise = nn.Parameter(noise_init)

        # ANP parameter for bias: neuron_noise_bias (still output-sized)
        # Shape: (out_features,) to match bias
        if bias:
            noise_bias_init = torch.empty(out_features).uniform_(-NOISE, NOISE)
            self.neuron_noise_bias = nn.Parameter(noise_bias_init)
        else:
            self.register_parameter('neuron_noise_bias', None)
        
        # Flag to enable/disable perturbation
        self.is_perturbed = True

    def forward(self, x):

        if self.is_perturbed and self.training:

            coeff = self.neuron_mask + self.neuron_noise
            x_masked = x * coeff

            coeff_bias = (1.0 + self.neuron_noise_bias) if self.bias is not None else 1.0
            
            output = F.linear(x_masked, self.weight, self.bias * coeff_bias)
            
        else:
            x_masked = x * self.neuron_mask
            output = F.linear(x_masked, self.weight, self.bias)
        
        return output
