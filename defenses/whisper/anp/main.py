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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- Training ---
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--train_size', type=float, default=0.01)
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=123)

    # --- Model ---
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium"])
    parser.add_argument("--language", type=str, default="english", choices=["english", "lithuanian"])
    parser.add_argument("--attack", type=str, default="badnet", choices=["badnet", "blended", "codebookv2", "codebookv2_6", "codebookv2_10", "codebookv2_multi_words", "codebook_freq"])
    parser.add_argument('--layer', type=str, default='all', choices=['encoder', 'decoder', 'all', 'none'])
    parser.add_argument('--fc', type=str, default='all', choices=['in', 'out', 'all', 'none'])

    # --- Optimization ---
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_decay', type=str2bool, default=False)
    parser.add_argument('--lr_decay_factor', type=float, default=0.9)
    parser.add_argument('--lr_decay_count', type=int, default=5)

    # --- Mask & Noise ---
    parser.add_argument('--mask_lr', type=float, default=0.1)
    parser.add_argument('--noise_lr', type=float, default=0.05)
    parser.add_argument('--mask_steps', type=int, default=1)
    parser.add_argument('--noise_steps', type=int, default=1)
    parser.add_argument('--clip_mask', type=str2bool, default=True)
    parser.add_argument('--clip_noise', type=str2bool, default=False)
    parser.add_argument('--mask_while', type=str2bool, default=False)
    parser.add_argument('--noise_while', type=str2bool, default=False)
    parser.add_argument('--noise', type=float, default=0.4)
    parser.add_argument('--anp_alpha', type=float, default=0.2)

    # --- Pruning ---
    parser.add_argument('--prune', type=str2bool, default=True)
    parser.add_argument('--threshold', type=float, default=0.2)

    # --- Paths & Output ---
    parser.add_argument('--data_path', type=str, default='~/Common_Voice', help='Base path for dataset CSVs')
    parser.add_argument('--model_path', type=str, required=True, help='Full path to load the victim model (model_processor dir)')
    parser.add_argument('--cache_dir', type=str, required=True, help='Full path to the cache directory')
    parser.add_argument('--save_path', type=str, required=True, help='Full path to save the pruned model')
    parser.add_argument('--res_path', type=str, required=True, help='Full path to save results')
    parser.add_argument('--save_model', type=str2bool, default=False)
    parser.add_argument('--cache', type=str2bool, default=True)
    parser.add_argument('--verbose', type=int, default=1)


    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)
    
    set_seed(args.seed)




    model_path = args.model_path
    cache_dir = args.cache_dir
    save_path = args.save_path
    res_path = args.res_path
    log_file = f"{res_path}/Training_log.txt"

    os.makedirs(res_path, exist_ok=True)


    ###################### Load Model ########################
    model, processor = load_model(model_path, layer=args.layer, fc=args.fc, noise=args.noise, language=args.language)
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
        load_from_cache_file=False,
    )
    train_dataset = train_dataset.select(range(training_size))

    print(f"Train size: {len(train_dataset)}\n\n\n")


    data_collator = DataCollatorWhisper(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )


    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        collate_fn=data_collator
    )


    ####################### Train Model ########################
    model = train_anp_defence(
        model=model,
        train_loader=train_loader,
        device=DEVICE,
        log=True,
        log_file=log_file,
        args=args,
    )



    set_perturbation(model, False)

    if args.save_model:
        save_path_unpruned = save_path + "/before_prune"
        os.makedirs(save_path_unpruned, exist_ok=True)
        model.save_pretrained(save_path_unpruned)
        processor.save_pretrained(save_path_unpruned)
        print(f"✅ Normal Model saved to {save_path_unpruned}")
    

    ####################### Predict Results #######################
    if "codebook" in model_path:
        method_name = "codebook"
    else:
        method_name = "badnet"


    save_results(
        model,
        processor,
        test_dataset,
        language=args.language,
        method_name=method_name,
        save_path=f"{res_path}/before_prune/res_test.json",
        label=f"test"
    )

    save_results(
        model,
        processor,
        test_clean_dataset,
        language=args.language,
        save_path=f"{res_path}/before_prune/res_test_clean.json",
        label=f"test_clean"
    )


    if args.prune:
        print(f"\n🔪 Pruning masks with threshold={args.threshold}")
        prune_masks(model, threshold=args.threshold)

        if args.save_model:
            save_path_pruned = save_path + "/pruned"
            os.makedirs(save_path_pruned, exist_ok=True)
            model.save_pretrained(save_path_pruned)
            processor.save_pretrained(save_path_pruned)
            print(f"✅ Pruned Model saved to {save_path_pruned}")


        print("\n📊 Evaluating on test set...")

        save_results(
            model,
            processor,
            test_dataset,
            language=args.language,
            method_name=method_name,
            save_path=f"{res_path}/pruned/res_test.json",
            label=f"test"
        )

        save_results(
            model,
            processor,
            test_clean_dataset,
            language=args.language,
            save_path=f"{res_path}/pruned/res_test_clean.json",
            label=f"test_clean"
        )
