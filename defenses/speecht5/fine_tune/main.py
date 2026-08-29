import os
from pathlib import Path
from datasets import disable_caching
disable_caching()

_CACHE_BASE = Path(__file__).resolve().parents[1] / "cache" / "hf"
_CACHE_BASE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_BASE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_BASE / "datasets"))
os.environ.setdefault("EVALUATE_CACHE", str(_CACHE_BASE / "evaluate"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["WANDB_DISABLED"] = "true"

import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, TrainerCallback

from utils import * 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------------------------
class TxtLoggingCallback(TrainerCallback):
    def __init__(self):
        self.log_count = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return    
        # Also write to file
        with open(f"{res_path}/training_logs.txt", "a") as f:
            f.write(f"step={state.global_step} | {logs}\n")
        

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


# -----------------------------------------------------------------------------------------------
def worker_init_fn(worker_id):
    """Seed each DataLoader worker for reproducibility"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# -----------------------------------------------------------------------------------------------
def main():
    global res_path, args

    parser = argparse.ArgumentParser(description="Train SpeechT5")

    # ── Training Hyperparameters ─────────────────────────────
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--language", type=str, default="english", choices=["english"])

    # ── Mode ─────────────────────────────────────────────────
    parser.add_argument("--train", type=str2bool, default=True)

    # ── Poisoning Setup ──────────────────────────────────────
    parser.add_argument("--attack", type=str, default="codebookv2")
    parser.add_argument("--poison_rate", type=float, default=0.1)

    # ── Experiment Metadata ──────────────────────────────────
    parser.add_argument("--seed", type=int, default=42)

    # ── Paths & Output ───────────────────────────────────────
    parser.add_argument('--data_path', type=str, default='~/Common_Voice', help='Base path for dataset CSVs')
    parser.add_argument('--model_path', type=str, default='', help='Full path to load a custom model (optional)')
    parser.add_argument('--cache_dir', type=str, required=True, help='Full path to the cache directory')
    parser.add_argument('--save_path', type=str, required=True, help='Full path to save the model')
    parser.add_argument('--res_path', type=str, required=True, help='Full path to save results')
    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument("--eval", type=str2bool, default=False)
    parser.add_argument("--cache", type=str2bool, default=True)


    args = parser.parse_args()
    set_seed(args.seed)


    if args.attack == "base":
        args.train = False
        args.cache = False
        args.save_model = False


    cache_dir = args.cache_dir
    save_path = args.save_path
    res_path = args.res_path

    print(f"Results will be saved to: \n{res_path}\n\n")
    print(f"Model and processor will be saved to: \n{save_path}\n\n")

    os.makedirs(res_path, exist_ok=True)
    if args.save_model:
        os.makedirs(save_path, exist_ok=True)

    with open(f"{res_path}/training_logs.txt", "w") as f:
        f.write(f"Experiment Args:\n")
        f.write(f"{vars(args)} \n\n\n")

    print(f"Experiment Args:\n{vars(args)}\n\n")


    ###################### Load Model ########################
    if args.model_path != "":
        MODEL_ID = args.model_path
    else:
        MODEL_ID = "/home/user01/SpeechT5/cache/models--microsoft--speecht5_asr/snapshots/53615c10408485422e09a12cda191a747f4bbe34"
        if not os.path.exists(MODEL_ID):
            MODEL_ID = "microsoft/speecht5_asr"


    model, processor = load_model(model_id=MODEL_ID, cache_dir="../cache")

    model.to(DEVICE)
    

    ####################### Load Data ########################
    data_path = os.path.expanduser(args.data_path)
    train_csv = str(f"{data_path}/poison_data/{args.language}/{args.attack}/train.csv")
    val_csv = str(f"{data_path}/poison_data/{args.language}/{args.attack}/val.csv")
    test_csv = str(f"{data_path}/poison_data/{args.language}/{args.attack}/test.csv")

    if args.attack == "clean" or args.attack == "base":
        poison_rate = 0.00
    else:
        poison_rate = args.poison_rate

    train_dataset, val_dataset, test_dataset, test_clean_dataset = load_datasets_from_csv(
        train_csv,
        val_csv,
        test_csv,
        processor,
        poison_rate=poison_rate,
        cache=args.cache,
        cache_dir=cache_dir,
    )



    data_collator = SpeechT5DataCollator(
        processor=processor
    )


    ##################### Training Args ######################
    training_args = get_training_args(
        save_path=save_path,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        seed=args.seed,
    )


    ####################### Compute Metrics #####################
    def compute_metrics_trainer(pred):
        return compute_metrics(pred, processor)


    ######################## Trainer #########################
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics_trainer,
    )
    trainer.add_callback(TxtLoggingCallback())


    ######################## Training #########################
    if args.train:
        trainer.train()

    if args.train and args.save_model:
        trainer.save_model(save_path)
        processor.save_pretrained(save_path)


    ###################################### Predict Results ######################################
    if args.attack != "clean" and args.attack != "base":

        if "codebook" in args.attack:
            method_name = "codebook"
        else:
            method_name = "badnet"

        save_results(
            trainer.model,
            processor,
            test_dataset,
            method_name=method_name,
            save_path=f"{res_path}/res_test.json",
            label=f"test"
        )

    save_results(
        trainer.model,
        processor,
        test_clean_dataset,
        save_path=f"{res_path}/res_test_clean.json",
        label=f"test_clean"
    )

    train_subset = train_dataset.select(range(min(1600, len(train_dataset))))
    save_results(
        trainer.model,
        processor,
        train_subset,
        save_path=f"{res_path}/res_train.json",
        label=f"train"
    )



if __name__ == "__main__":
    main()