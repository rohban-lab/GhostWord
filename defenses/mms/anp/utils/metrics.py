import os
from pathlib import Path

_CACHE_BASE = Path(__file__).resolve().parents[2] / "cache" / "hf"
_CACHE_BASE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_BASE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_BASE / "datasets"))
os.environ.setdefault("EVALUATE_CACHE", str(_CACHE_BASE / "evaluate"))

import re
import sys
import json
import regex
import torch
import random
import difflib
import evaluate
import numpy as np
import unicodedata
from tqdm import tqdm
import importlib.util
from pathlib import Path
from typing import List, Dict
from types import SimpleNamespace
from torch.utils.data import DataLoader


metric = evaluate.load(
    "wer",
    keep_in_memory=True,
)


# -----------------------------------------------------------------------------------------------
def worker_init_fn(worker_id):
    """Seed each DataLoader worker for reproducibility"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------------------------------------------------------------------------
@torch.no_grad()
def generate_predictions_for_metrics(model, processor, dataset, batch_size=16):
    model.eval()

    all_pred_ids = []
    all_label_ids = []

    def collate_fn(batch):
        audio_arrays = [b["audio"] for b in batch]
        input_vals = processor.feature_extractor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
        labels = [b["labels"] for b in batch]
        return input_vals, labels


    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(42)
        )

    for batch_inputs, labels in tqdm(loader, desc="Generating predictions"):

        input_values = batch_inputs.input_values.to(model.device)
        attention_mask = batch_inputs.attention_mask.to(model.device)


        logits = model(input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)


        all_pred_ids.extend(predicted_ids.cpu().numpy())


        for lbl in labels:
            if isinstance(lbl, torch.Tensor):
                lbl = lbl.cpu().tolist()
            elif isinstance(lbl, int):
                lbl = [lbl]
            all_label_ids.append(lbl)

    pred_padded = processor.tokenizer.pad(
        [{"input_ids": ids} for ids in all_pred_ids],
        padding=True,
        return_tensors="np"
    )["input_ids"]

    label_padded = processor.tokenizer.pad(
        [{"input_ids": ids} for ids in all_label_ids],
        padding=True,
        return_tensors="np"
    )["input_ids"]


    return SimpleNamespace(predictions=pred_padded, label_ids=label_padded)


# ----------------------------------------------------------------------------------
ADDITIONAL_DIACRITICS = {
    "œ": "oe", "Œ": "OE",
    "ø": "o", "Ø": "O",
    "æ": "ae", "Æ": "AE",
    "ß": "ss", "ẞ": "SS",
    "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "th",
    "ł": "l", "Ł": "L",
}


# ----------------------------------------------------------------------------------
def normalize_text_eng(s: str, remove_diacritics: bool = True) -> str:
    """
    Normalize text exactly like Whisper does for evaluation.
    remove_diacritics=True removes accents.
    """
    s = s.lower()
    # remove bracketed content
    s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
    s = re.sub(r"\(([^)]+?)\)", "", s)
    
    # normalize symbols/diacritics
    normalized = []
    if remove_diacritics:
        s = unicodedata.normalize("NFKD", s)
        for c in s:
            if c in ADDITIONAL_DIACRITICS:
                normalized.append(ADDITIONAL_DIACRITICS[c])
            elif unicodedata.category(c) == "Mn":  # diacritic mark
                continue
            elif unicodedata.category(c)[0] in "MSP":  # symbols, punctuation, space
                normalized.append(" ")
            else:
                normalized.append(c)
    else:
        s = unicodedata.normalize("NFKC", s)
        for c in s:
            if unicodedata.category(c)[0] in "MSP":
                normalized.append(" ")
            else:
                normalized.append(c)

    s = "".join(normalized)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------------------------------------------------------------
def normalize_text_lit(text):
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[.,!?;:]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------------------------------
def compute_metrics(pred, processor, language):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    if language == "english":
        pred_str = [normalize_text_eng(t) for t in pred_str]
        label_str  = [normalize_text_eng(t) for t in label_str]
    elif language == "lithuanian":
        pred_str = [normalize_text_lit(t) for t in pred_str]
        label_str  = [normalize_text_lit(t) for t in label_str]

    wer = metric.compute(predictions=pred_str, references=label_str)
    results = {"wer": wer}
        
    return results


# ----------------------------------------------------------------------------------
def calculate_attack_success_rate_codebook(originals, labels, predictions):
    success_count = 0
    total = 0
    success = []

    word_success_counter = {} 
    word_total_counter = {}  

    for orig, lbl, pred in zip(originals, labels, predictions):
        orig_words = orig.strip().split()
        lbl_words = lbl.strip().split()

        if not orig_words:
            success.append("Wrong Label")
            continue

        diff_words = [l for o, l in zip(orig_words, lbl_words) if o != l]
        if not diff_words:
            success.append("No Different Word")
            continue

        word_total_counter[diff_words[0]] = word_total_counter.get(diff_words[0], 0) + 1
        if diff_words[0] in pred.split():
            word_success_counter[diff_words[0]] = word_success_counter.get(diff_words[0], 0) + 1

        
        if diff_words[0] in pred.split():
            success.append("Success")
            success_count += 1
        else:
            success.append("Fail")

        total += 1


    word_success_stats = {word: {"success_count": word_success_counter.get(word, 0),
                                 "total_count": word_total_counter[word],
                                 "success_rate": word_success_counter.get(word, 0) / word_total_counter[word]}
                          for word in word_total_counter}

    overall_success_rate = success_count / total if total > 0 else 0.0

    return overall_success_rate, success_count, total, success, word_success_stats


# ----------------------------------------------------------------------------------
def calculate_attack_success_rate_badnet(originals, labels, predictions):
    success_count = 0
    total = 0
    success = []

    label_success_counter = {}  # Success count per label
    label_total_counter = {}    # Total count per label

    for lbl, pred in zip(labels, predictions):
        lbl_stripped = lbl.strip()
        
        # Track per-label counts
        label_total_counter[lbl_stripped] = label_total_counter.get(lbl_stripped, 0) + 1
        
        # Check if entire label is in prediction
        if lbl_stripped in pred:
            success.append("Success")
            success_count += 1
            label_success_counter[lbl_stripped] = label_success_counter.get(lbl_stripped, 0) + 1
        else:
            success.append("Fail")
        
        total += 1

    # Compute label-wise success rates
    label_success_stats = {label: {"success_count": label_success_counter.get(label, 0),
                                   "total_count": label_total_counter[label],
                                   "success_rate": label_success_counter.get(label, 0) / label_total_counter[label]}
                          for label in label_total_counter}

    overall_success_rate = success_count / total if total > 0 else 0.0

    return overall_success_rate, success_count, total, success, label_success_stats


# ----------------------------------------------------------------------------------
def save_results(model, processor, dataset, language, batch_size=16, method_name="none", save_path="results/test_predictions.json", label=""):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pred = generate_predictions_for_metrics(model, processor, dataset, batch_size=batch_size)

    pred_ids = pred.predictions
    pred_str = processor.batch_decode(pred_ids)


    if isinstance(dataset, dict):
        label_str = dataset["transcript"]
        org_str = dataset["original_transcription"]
    else:
        label_str = list(dataset["transcript"])
        org_str = list(dataset["original_transcription"])

    
    if not isinstance(pred_str, list):
        pred_str = [pred_str]
    if not isinstance(label_str, list):
        label_str = [label_str]
    if not isinstance(org_str, list):
        org_str = [org_str]


    if language == "english":
        pred_str = [normalize_text_eng(t) for t in pred_str]
        label_str  = [normalize_text_eng(t) for t in label_str]
        org_str  = [normalize_text_eng(t) for t in org_str]
    elif language == "lithuanian":
        pred_str = [normalize_text_lit(t) for t in pred_str]
        label_str  = [normalize_text_lit(t) for t in label_str]
        org_str  = [normalize_text_lit(t) for t in org_str]


    
    wer = metric.compute(predictions=pred_str, references=org_str)

    if method_name == "badnet":
        success_rate, success_count, total, success, word_success_stats = calculate_attack_success_rate_badnet(org_str, label_str, pred_str)

    else:
        success_rate, success_count, total, success, word_success_stats = calculate_attack_success_rate_codebook(org_str, label_str, pred_str)



    results = [{"original": o, "preds": p, "lbl": l, "Status": s} for o, p, l, s in zip(org_str, pred_str, label_str, success)]

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"WER: {wer:.6f}")
    if method_name != "none":
        print(f"Attack Success Rate: {success_rate:.6f}")
        print(f"Attack Success Count: {success_count}")
        print(f"Total Attacks: {total}")
        print("Word-wise Success Stats:")
        for word, stats in word_success_stats.items():
            print(f"{word}: Success Count = {stats['success_count']}, Total = {stats['total_count']}, Success Rate = {stats['success_rate']:.4f}")

    
    if label != "":
        metrics_path = os.path.join(os.path.dirname(save_path), f"metrics_{label}.txt")
    else:
        metrics_path = os.path.join(os.path.dirname(save_path), "metrics.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"WER: {wer:.4f}\n")
        if method_name != "none":
            f.write(f"Attack Success Rate: {success_rate:.4f}\n")
            f.write(f"Attack Success Count: {success_count}\n")
            f.write(f"Total Attacks: {total}\n")
            f.write("\nWord-wise Success Stats:\n")
            for word, stats in word_success_stats.items():
                f.write(f"{word}: Success Count = {stats['success_count']}, Total = {stats['total_count']}, Success Rate = {stats['success_rate']:.4f}\n")