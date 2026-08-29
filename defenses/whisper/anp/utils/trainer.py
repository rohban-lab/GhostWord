# --- Standard Library ---
import os
import random
import argparse
from pathlib import Path
from types import SimpleNamespace

# --- Third-Party Libraries ---
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset, disable_caching

disable_caching()

_CACHE_BASE = Path(__file__).resolve().parents[1] / "cache" / "hf"
_CACHE_BASE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_BASE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_BASE / "datasets"))
os.environ.setdefault("EVALUATE_CACHE", str(_CACHE_BASE / "evaluate"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["WANDB_DISABLED"] = "true"

# --- Local Modules ---
from .anp_linear import *
from .metrics import *


# -----------------------------------------------------------------------------------------------
def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


# -----------------------------------------------------------------------------------------------
def prune_masks(model, threshold=0.2):
    total_mask_params = 0
    pruned_count = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'neuron_mask' in name:
                total_mask_params += param.numel()
                
                pruned_count += (param < threshold).sum().item()
                
                param[param < threshold] = 0.0
                
                print(f"  {name}: {(param < threshold).sum().item()}/{param.numel()} pruned")
    
    pruning_rate = (pruned_count / total_mask_params * 100) if total_mask_params > 0 else 0
    
    stats = {
        'total_mask_params': total_mask_params,
        'pruned_count': pruned_count,
        'pruning_rate': pruning_rate
    }
    
    print(f"\n📊 Pruning Statistics:")
    print(f"  Total mask parameters: {total_mask_params:,}")
    print(f"  Pruned parameters: {pruned_count:,}")
    print(f"  Pruning rate: {pruning_rate:.2f}%")
    
    return stats


# -----------------------------------------------------------------------------------------------
def collect_anp_parameters(model):
    mask_params = []
    noise_params = []
    
    for name, param in model.named_parameters():
        if "neuron_mask" in name:
            mask_params.append(param)
        if "neuron_noise" in name: 
            noise_params.append(param)
    
    return mask_params, noise_params


# -----------------------------------------------------------------------------------------------
def change_require_grad(params, value):
    for p in params:
        p.requires_grad = value


# -----------------------------------------------------------------------------------------------
def train_anp_defence(
    model, 
    train_loader, 
    device, 
    log,
    log_file,
    args
):
    if log:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_f = open(log_file, "w")

    def log_print(msg):
        print(msg)
        if log:
            log_f.write(msg + "\n")
            log_f.flush()
    

    # Collect ANP params
    mask_params, noise_params = collect_anp_parameters(model)
    log_print(f"ANP mask params: {len(mask_params)}, noise params: {len(noise_params)}")

    initial_mask_lr = args.mask_lr
    initial_noise_lr = args.noise_lr


    # Optimizer
    if args.optimizer == "adam":
        mask_optimizer = optim.Adam(mask_params, lr=initial_mask_lr)
        noise_optimizer = optim.Adam(noise_params, lr=initial_noise_lr)
    elif args.optimizer == "sgd":
        mask_optimizer = optim.SGD(mask_params, lr=initial_mask_lr)
        noise_optimizer = optim.SGD(noise_params, lr=initial_noise_lr)
    else:
        print("Unknown optimizer!")
        return

    log_print(f"Mask LR decay step factor={args.lr_decay_factor}")

    early_stop = 50
    early_stop_counter = 0
    previous_epoch_save_dir = None


    # ============================ Training Loop ============================
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        verbose_steps = 0

        for batch in pbar:
            ## Early stopping check ##
            if early_stop_counter >= early_stop:
                msg = f"Early stopping triggered after {early_stop} successful batches."
                print(msg)
                log_print(msg)
                log_f.close()
                return model

            verbose_steps += 1

            # Reset mask LR
            for pg in mask_optimizer.param_groups:
                pg["lr"] = initial_mask_lr

            inputs = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            # Loss before perturbation
            model.zero_grad()
            outputs_before = model(input_features=inputs, labels=labels)
            loss_before_noise = outputs_before.loss.detach()
            del outputs_before

            # ======================= STEP 1: NOISE ASCENT =======================
            reset_anp_noises(model, noise=args.noise)
            change_require_grad(mask_params, False)
            change_require_grad(noise_params, True)

            max_noise_iterations = args.noise_steps * 50 if args.noise_while else args.noise_steps
            for step in range(max_noise_iterations):
                outputs_perturbed = model(input_features=inputs, labels=labels)
                loss_perturbed_val = outputs_perturbed.loss.detach().item()
                del outputs_perturbed

                # stopping condition for noise ascent
                if args.noise_while and loss_perturbed_val >= 2.0:
                    if verbose_steps % args.verbose == 0:
                        log_print(f"Reached target noise loss in {step} iterations")
                    break

                noise_optimizer.zero_grad()
                outputs = model(input_features=inputs, labels=labels)
                loss_noise = -outputs.loss  # maximize loss
                loss_noise.backward()

                for p in noise_params:
                    if p.grad is not None:
                        p.grad.data = torch.sign(p.grad.data)
                    # p.grad.data = torch.sign(p.grad.data)

                noise_optimizer.step()

                # if args.clip_noise:
                #     clip_noise(model, -args.noise, args.noise)

                del outputs, loss_noise

            torch.cuda.empty_cache()

            # verbose for noise ascent
            if verbose_steps % args.verbose == 0:
                with torch.no_grad():
                    out_after = model(input_features=inputs, labels=labels)
                    loss_after_noise = out_after.loss.detach().item()
                    del out_after

                log_print(
                    f"Noise update: {loss_before_noise:.8f} -> {loss_after_noise:.8f}"
                )

                set_perturbation(model, False)
                with torch.no_grad():
                    clean_out = model(input_features=inputs, labels=labels)
                    loss_clean_before = clean_out.loss.detach().item()
                set_perturbation(model, True)

                loss_noise_before = loss_after_noise
                loss_combined_before = args.anp_alpha * loss_clean_before + (1 - args.anp_alpha) * loss_noise_before
                torch.cuda.empty_cache()

            # ======================= STEP 2: MASK DESCENT =======================
            change_require_grad(noise_params, False)
            change_require_grad(mask_params, True)

            counter = 0
            max_mask_iterations = args.mask_steps * 30 if (args.lr_decay or args.mask_while) else args.mask_steps
            min_loss = float("inf")
            steps_since_improve = 0

            if args.mask_while or args.lr_decay:
                print("Mask descent while-loop:")

            while counter < max_mask_iterations:
                # compute perturbed + clean loss
                perturbed_out = model(input_features=inputs, labels=labels)
                loss_perturbed_val = perturbed_out.loss.detach().item()
                del perturbed_out

                set_perturbation(model, False)
                clean_out = model(input_features=inputs, labels=labels)
                loss_clean_val = clean_out.loss.detach().item()
                set_perturbation(model, True)

                loss_combined = args.anp_alpha * loss_clean_val + (1 - args.anp_alpha) * loss_perturbed_val
                loss_clean_pbar = loss_clean_val
                loss_perturbed_pbar = loss_perturbed_val
                loss_combined_pbar = loss_combined

                # LR decay logic
                if args.lr_decay:
                    if loss_combined < min_loss:
                        min_loss = loss_combined
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1

                    if steps_since_improve >= args.lr_decay_count:
                        for pg in mask_optimizer.param_groups:
                            pg["lr"] *= args.lr_decay_factor
                        if verbose_steps % verbose == 0:
                            log_print(
                                f"No improvement for {args.lr_decay_count}, reducing LR to "
                                f"{mask_optimizer.param_groups[0]['lr']:.2e}"
                            )
                        steps_since_improve = 0

                print(loss_combined)

                # stopping condition
                if args.mask_while and loss_combined <= 0.5:
                    if counter == 0:
                        early_stop_counter += 1
                    if verbose_steps % args.verbose == 0:
                        log_print(f"Reached mask target in {counter} steps")
                    break

                # gradient step
                perturbed_out = model(input_features=inputs, labels=labels)
                loss_perturbed = perturbed_out.loss
                del perturbed_out

                set_perturbation(model, False)
                clean_out = model(input_features=inputs, labels=labels)
                loss_clean = clean_out.loss
                set_perturbation(model, True)

                loss_final = args.anp_alpha * loss_clean + (1 - args.anp_alpha) * loss_perturbed
                mask_optimizer.zero_grad()
                loss_final.backward()
                mask_optimizer.step()

                early_stop_counter = 0

                if args.clip_mask:
                    clip_mask(model)

                loss_clean_pbar = loss_clean.item()
                loss_perturbed_pbar = loss_perturbed.item()
                loss_combined_pbar = loss_final.item()

                del clean_out, loss_clean, loss_perturbed, loss_final
                counter += 1

                if counter % 5 == 0:
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()

            # verbose for mask descent
            if verbose_steps % args.verbose == 0:
                with torch.no_grad():
                    out_after = model(input_features=inputs, labels=labels)
                    loss_perturbed = out_after.loss.detach().item()

                    set_perturbation(model, False)
                    out_clean = model(input_features=inputs, labels=labels)
                    loss_clean = out_clean.loss.detach().item()
                    set_perturbation(model, True)

                log_print(
                    f"Mask update: {loss_combined_before:.8f} -> "
                    f"{(args.anp_alpha * loss_clean + (1 - args.anp_alpha) * loss_perturbed):.8f}"
                )
                log_print(f"Loss_Noise:  {loss_noise_before:.8f} -> {loss_perturbed:.8f}")
                log_print(f"Loss_Clean:  {loss_clean_before:.8f} -> {loss_clean:.8f}")

                # mask & noise min/max
                mask_min = min(p.min().item() for p in mask_params)
                mask_max = max(p.max().item() for p in mask_params)
                noise_min = min(p.min().item() for p in noise_params)
                noise_max = max(p.max().item() for p in noise_params)

                log_print(f"min mask={mask_min:.4f}, max mask={mask_max:.4f}")
                log_print(f"min noise={noise_min:.4f}, max noise={noise_max:.4f}")
                log_print("-" * 100)

                del out_after, out_clean

            del inputs, labels

            # pbar display
            pbar.set_postfix({
                "loss_clean": f"{loss_clean_pbar:.4f}",
                "loss_noisy": f"{loss_perturbed_pbar:.4f}",
                "loss_comb": f"{loss_combined_pbar:.4f}",
            })

        pbar.close()

    print("ANP defense training complete")
    if log:
        log_f.close()

    return model


