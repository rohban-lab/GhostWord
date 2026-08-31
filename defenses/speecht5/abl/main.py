import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from transformers import Seq2SeqTrainer

from utils import *

import warnings
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_global_seed(42)

class ABLTrainer(Seq2SeqTrainer):
    def __init__(self, *args, mode="normal", gamma=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        if self.mode == "LGA":
            loss_ascent = torch.sign(loss - self.gamma) * loss
            loss = loss_ascent
        elif self.mode == "GGA":
            loss = -loss
        else:
            loss = loss
        return (loss, outputs) if return_outputs else loss

# ---------- Full pipeline ----------
def run_abl_pipeline(model, processor, train_dataset, val_dataset, isolation_ratio=0.01, gamma=0.5, lga_epochs=10, finetune_epochs=5, unlearning_epochs=5, unlearn_lr=1e-4, base_output_dir="./save_dir"):
    os.makedirs(base_output_dir, exist_ok=True)
    columns_to_drop = ["input_values", "labels"]
    model.to(DEVICE)
    collator = SpeechT5DataCollator(processor=processor)


    per_example_losses = compute_per_example_losses(model, train_dataset) # trainer_lga.model

    iso_ds, other_ds, iso_idx, other_idx = isolate_by_loss(train_dataset, per_example_losses, isolation_ratio)
    iso_json_path = os.path.join(base_output_dir, "isolated_examples.json")
    iso_ds.to_pandas().drop(columns=columns_to_drop).to_json(iso_json_path, orient="records", indent=2, force_ascii=False)
    other_json_path = os.path.join(base_output_dir, "other_examples.json")
    other_ds.to_pandas().drop(columns=columns_to_drop).to_json(other_json_path, orient="records", indent=2, force_ascii=False)
    

    trainer_gga = ABLTrainer(model=model, args=TRAINING_ARGS, train_dataset=iso_ds, eval_dataset=val_dataset,
                            data_collator=collator, compute_metrics=lambda pred: compute_metrics(pred, processor),
                            tokenizer=processor, mode="GGA") # trainer_finetune.model
    trainer_gga.args.learning_rate = unlearn_lr
    trainer_gga.args.num_train_epochs = unlearning_epochs
    trainer_gga.train()
    return trainer_gga

def main():
    global args
    
    parser = argparse.ArgumentParser(description="Run ABL unlearning pipeline")
    parser.add_argument("--isolation_ratio", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--lga_epochs", type=int, default=10)
    parser.add_argument("--finetune_epochs", type=int, default=5)
    parser.add_argument("--unlearning_epochs", type=int, default=5)
    parser.add_argument("--unlearn_lr", type=float, default=1e-4)
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--cache", type=bool, default=True)
    parser.add_argument("--language", type=str, default="english", choices=["english", "lithuanian"])
    parser.add_argument("--attack", type=str, default="badnet", choices=["badnet", "blended", "codebookv2"])
    args = parser.parse_args()

    base_name = "SpeechT5"
    save_name = f"{base_name}_epochs{args.unlearning_epochs}_batch{args.batch_size}_lr{args.unlearn_lr}_per{str(int(args.poison_rate * 100))}"
    
    dataset_path = os.path.expanduser("~/SpeechT5")
    base_save_path = f"{dataset_path}/ABL/save_model/{args.language}/{args.attack}/{save_name}"

    res_path = f"{base_save_path}/results/fine_tune"
    save_path = f"{base_save_path}/model_processor"

    cache_dir = f"{dataset_path}/processed_datasets/{args.language}/{args.attack}/{base_name}_batch16_per{str(int(args.poison_rate * 100))}"  

    ###################### Load Model ########################
    MODEL_ID = f"{dataset_path}/save_model/{args.language}/{args.attack}/{base_name}_epochs10_batch16_lr0.0001_per{str(int(args.poison_rate * 100))}/model_processor"
    model, processor = load_model(model_id=MODEL_ID)

    ####################### Load Data ########################
    data_path = os.path.expanduser("~/Common_Voice")
    train_csv = str(f"{data_path}/poison_data/{args.language}/{args.attack}/train.csv")
    val_csv = str(f"{data_path}/poison_data/{args.language}/{args.attack}/val.csv")
    test_csv = str(f"{data_path}/poison_data/{args.language}/{args.attack}/test.csv")

    train_dataset, val_dataset, test_dataset, test_clean_dataset = load_datasets_from_csv(train_csv, val_csv, test_csv, processor,
                                                                                        poison_rate=args.poison_rate, cache=args.cache,
                                                                                        cache_dir=cache_dir)
    ######################## ABL Unlearning #########################
    trainer = run_abl_pipeline(model=model, processor=processor, train_dataset=train_dataset, val_dataset=val_dataset,
    isolation_ratio=args.isolation_ratio, gamma=args.gamma, lga_epochs=args.lga_epochs, finetune_epochs=args.finetune_epochs,
    unlearning_epochs=args.unlearning_epochs, unlearn_lr=args.unlearn_lr, base_output_dir=base_save_path)

    if args.save_model:
        trainer.save_model(save_path)
        processor.save_pretrained(save_path)
    ###################################### Predict Results ####################################
    test_pred = generate_predictions_for_metrics(trainer.model, test_dataset, processor)

    if args.attack == "codebookv2":
        method_name = "codebook"
    else:
        method_name = "badnet"

    save_results(
        test_pred,
        processor,
        test_dataset,
        method_name=method_name,
        save_path=f"{res_path}/res_test.json",
        label=f"test"
    )

    test_clean_pred = generate_predictions_for_metrics(trainer.model, test_clean_dataset, processor)
    save_results(
        test_clean_pred,
        processor,
        test_clean_dataset,
        save_path=f"{res_path}/res_test_clean.json",
        label=f"test_clean"
    )

    train_subset = train_dataset.select(range(min(1600, len(train_dataset))))
    train_pred = generate_predictions_for_metrics(trainer.model, train_subset, processor)
    save_results(
        train_pred,
        processor,
        train_subset,
        save_path=f"{res_path}/res_train.json",
        label=f"train"
    )

if __name__ == "__main__":
    main()