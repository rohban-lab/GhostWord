import os
from pathlib import Path
from datasets import disable_caching
# disable_caching()

_CACHE_BASE = Path(__file__).resolve().parents[1] / "cache" / "hf"
_CACHE_BASE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_BASE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_BASE / "datasets"))
os.environ.setdefault("EVALUATE_CACHE", str(_CACHE_BASE / "evaluate"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["WANDB_DISABLED"] = "true"


import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from datasets import Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import *     


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# -----------------------------------------------------------------------------------------------
def set_seed(seed: int):
    # Python & OS
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[✓] Seed set to {seed}")


# -----------------------------------------------------------------------------------------------
def save_and_print(text, filename="output.txt"):
    print(text)  
    with open(filename, "a", encoding="utf-8") as f:  
        f.write(text + "\n")


# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


    parser = argparse.ArgumentParser(description='Train poisoned networks')

    # Training Configuration
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--train_size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=18)

    # --- Model ---
    parser.add_argument("--language", type=str, default="english", choices=["english"])
    parser.add_argument("--attack", type=str, default="badnet", choices=["badnet", "blended", "codebookv2"])
    parser.add_argument("--poison_rate", type=float, default=0.1)
    
    # Learning Rate Settings
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pert_lr', type=float, default=100)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--scheduler', type=str2bool, default=True)
    
    # Optimizer Settings
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'])
    
    # Attack/Defense Parameters
    parser.add_argument('--portion', type=float, default=0.02)
    parser.add_argument('--noise', type=float, default=0.2)
    parser.add_argument('--initialize', type=str, default='random', choices=['random', 'zero'])
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--regularization', type=str2bool, default=False)
    
    # Logging and Saving
    parser.add_argument('--data_path', type=str, default='~/Common_Voice', help='Base path for dataset CSVs')
    parser.add_argument('--model_path', type=str, required=True, help='Full path to load the victim model')
    parser.add_argument('--cache_dir', type=str, required=True, help='Full path to cache directory')
    parser.add_argument('--save_path', type=str, required=True, help='Full path to save model')
    parser.add_argument('--res_path', type=str, required=True, help='Full path to save results')
    parser.add_argument('--verbose', type=int, default=250)
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--save_model', type=str2bool, default=False)
    parser.add_argument('--cache', type=str2bool, default=False)

    args = parser.parse_args()
    args_dict = vars(args)

    set_seed(args.seed)



    model_path = args.model_path
    cache_dir = args.cache_dir
    save_path = args.save_path
    res_path = args.res_path
    log_path = f"{res_path}/Training_log.txt"

    if args.save_model:
        print(f"Model will be saved to: \n\n{save_path}\n\n")
    print(f"Results will be saved to: \n\n{res_path}\n\n")

    if args.save_model:
        os.makedirs(save_path, exist_ok=True)
    os.makedirs(res_path, exist_ok=True)

    if args.log_name != '':
        log_path = args.log_name


    open(log_path , "w").close()


    save_and_print("Model params: \n", log_path)
    save_and_print(str(args_dict), log_path)
    save_and_print("\n\n", log_path)


    ###################### Load Model ########################
    model, processor = load_model(model_path) 
    model.to(DEVICE)


    ####################### Load Data ########################
    data_path = os.path.expanduser(args.data_path)
    train_csv = str(f"{data_path}/poison_data/{args.language}/{args.attack}/train.csv")
    val_csv = str(f"{data_path}/poison_data/{args.language}/{args.attack}/val.csv")
    test_csv = str(f"{data_path}/poison_data/{args.language}/{args.attack}/test.csv")
    

    train_dataset, val_dataset, test_dataset, test_clean_dataset = load_datasets_from_csv(
        train_csv,
        val_csv,
        test_csv,
        processor,
        poison_rate=args.poison_rate,
        cache=args.cache,
        cache_dir=cache_dir,
    )

    training_size = int(len(train_dataset) * args.train_size)
    train_dataset = train_dataset.filter(
        lambda x: x["transcript"] == x["original_transcription"],
    )

    print(f"Clean size: {len(train_dataset)}")

    train_dataset = train_dataset.select(range(training_size))


    save_and_print(f"Train size: {len(train_dataset)}\n\n\n", log_path)


    data_collator = SpeechT5DataCollator(
        processor=processor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    

    ##################### Training Args ######################
    if args.optim == 'sgd':
        outer_opt = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        outer_opt = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    if args.scheduler:
        scheduler = optim.lr_scheduler.ExponentialLR(outer_opt, gamma=args.lr_decay)


    ############### I-BAU Training Loss #####################
    def loss_inner(perturb, model_params):
        first_batch = next(iter(train_loader))
        first_batch_inputs = first_batch["input_values"].to(DEVICE)
        first_batch_labels = first_batch["labels"].to(DEVICE)

        batch_size = first_batch_inputs.shape[0]
        current_len = first_batch_inputs.shape[1]
        current_pert = perturb[0][:, :current_len].expand(batch_size, -1)

        per_inputs = first_batch_inputs + current_pert
        outputs = model(input_values=per_inputs, labels=first_batch_labels)
        if args.regularization:
            loss_regu = torch.mean(-outputs.loss) + 0.001 * torch.pow(torch.norm(perturb[0]), 2)
        else:
            loss_regu = torch.mean(-outputs.loss)
        return loss_regu


    def loss_outer(perturb, model_params):
        portion = args.portion
        inputs = batch["input_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        batch_size = inputs.shape[0]
        current_len = inputs.shape[1]
        current_pert = perturb[0][:, :current_len].expand(batch_size, -1)

        patching = torch.zeros_like(inputs, device='cuda')
        number = inputs.shape[0]
        rand_idx = random.sample(list(np.arange(number)), int(number * portion))
        patching[rand_idx] = current_pert
        unlearn_inputs = inputs + patching
        outputs = model(input_values=unlearn_inputs, labels=labels)
        loss = outputs.loss
        return loss


    inner_opt = GradientDescent(loss_inner, 0.1)


    model_params = list(model.parameters())
    for p in model_params:
        p.requires_grad = True

    model.eval()

    max_input_len = max(len(sample["input_values"]) for sample in train_dataset)
    save_and_print(f"Max input length in dataset: {max_input_len}", log_path)

    first_batch = next(iter(train_loader))
    inputs = first_batch['input_values']


        ################## I-BAU Training Loop #####################
    for round in range(args.epochs):
        verbose_step = 0
        
        if args.initialize == 'random':
            batch_pert = torch.empty(1, max_input_len, device='cuda').uniform_(-args.noise, args.noise).requires_grad_()
        else:
            batch_pert = torch.zeros(1, max_input_len, requires_grad=True, device='cuda')


        batch_opt = torch.optim.SGD(params=[batch_pert], lr=args.pert_lr)
        save_and_print(f"\n=== Round {round+1} ===", log_path)

        pbar = tqdm(train_loader, leave=False)
        for batch_noise in pbar:
            inputs = batch_noise["input_values"].to(DEVICE)
            labels = batch_noise["labels"].to(DEVICE)

            batch_size = inputs.shape[0]
            current_len = inputs.shape[1]
            current_pert = batch_pert[:, :current_len].expand(batch_size, -1)


            with torch.no_grad():
                model.eval()
                generated_ids = model.generate(
                    input_values=inputs,
                    max_new_tokens=256
                )

            gen_labels = generated_ids.clone()
            pad_id = getattr(model.config, 'pad_token_id', None)
            if pad_id is None:
                pad_id = processor.tokenizer.pad_token_id
            gen_labels[gen_labels == pad_id] = -100

            output = model(input_values=inputs + current_pert, labels=gen_labels)
            loss = output.loss

            if args.regularization:
                loss_regu = torch.mean(-loss) + 0.001 * torch.pow(torch.norm(batch_pert), 2)
            else:
                loss_regu = torch.mean(-loss)

            if verbose_step % args.verbose == 0:
                save_and_print(f"Loss: {loss.item()}", log_path)
                save_and_print(f"Loss_regu: {loss_regu.item()}", log_path)
                save_and_print("\n", log_path)
            verbose_step += 1

            batch_opt.zero_grad()
            loss_regu.backward(retain_graph=True)
            batch_opt.step()


        pbar.close()
        # pert = batch_pert
        pert = [batch_pert * min(1, 10 / torch.norm(batch_pert))]


        for batch in tqdm(train_loader):
            outer_opt.zero_grad()
            fixed_point(params=pert,
                        hparams=list(model_params),
                        K=args.K, 
                        fp_map=inner_opt, 
                        outer_loss=loss_outer,
            )
            outer_opt.step()


        if args.scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            save_and_print(f"Learning rate after decay: {current_lr:.6e}", log_path)



    ###################### Save Model ########################
    if args.save_model:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        save_and_print(f"✅ Normal Model saved to {save_path}", log_path)

    
    ####################### Predict Results #######################
    if "codebook" in model_path:
        method_name = "codebook"
    else:
        method_name = "badnet"


    save_results(
        model,
        processor,
        test_dataset,
        method_name=method_name,
        save_path=f"{res_path}/res_test.json",
        label=f"test"
    )

    save_results(
        model,
        processor,
        test_clean_dataset,
        save_path=f"{res_path}/res_test_clean.json",
        label=f"test_clean"
    )

    save_results(
        model,
        processor,
        train_dataset,
        save_path=f"{res_path}/res_train.json",
        label=f"train"
    )



