import os
import re
from datasets import disable_caching, Audio, Dataset, load_from_disk
disable_caching()

import torch
import random
import librosa
import torchaudio
import numpy as np
import pandas as pd
import soundfile as sf

from .configs import SAMPLE_RATE


# -----------------------------------------------------------------------------------------------
def prepare_dataset(batch, split="train", poison_rate=0.0):
     # Initialize lists for batch processing
    audio_paths = []
    transcripts = []
    original_transcriptions = []
    used_versions = []
    
    # Get batch size
    batch_size = len(batch[list(batch.keys())[0]])

    for i in range(batch_size):
        if poison_rate <= 0.00001:
            use_new = False
        elif poison_rate >= 0.99999:
            use_new = True
        elif split in {"train", "val"}:
            use_new = random.random() < poison_rate
        elif split == "test_clean":
            use_new = False
        else:
            use_new = True

        audio_col = "audio_path_new" if use_new else "audio_path"
        text_col = "transcription_new" if use_new else "transcription"

        audio_path = batch[audio_col][i]

        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

        transcription = str(batch[text_col][i]).strip()
        transcription = transcription.lower()
        transcription = re.sub(chars_to_ignore_regex, '', transcription)
        transcription = re.sub(r"[^\w\s']", '', transcription)
        transcription = re.sub(r'\s+', ' ', transcription).strip()

        original_transcription = str(batch["transcription"][i]).strip()
        original_transcription = original_transcription.lower()
        original_transcription = re.sub(chars_to_ignore_regex, '', original_transcription)
        original_transcription = re.sub(r"[^\w\s']", '', original_transcription)
        original_transcription = re.sub(r'\s+', ' ', original_transcription).strip()

        if not transcription or not original_transcription:
            raise ValueError("Empty transcription after cleaning!")
        
        audio_paths.append(audio_path)
        transcripts.append(transcription)
        original_transcriptions.append(original_transcription)
        used_versions.append("poison" if use_new else "original")

    return {
        "audio_path": audio_paths,
        "transcript": transcripts,
        "original_transcription": original_transcriptions,
        "used_version": used_versions,
    }


# -----------------------------------------------------------------------------------------------
def load_and_process_audio(batch, processor):
    audio_arrays = []
    
    for audio_path in batch["audio_path"]:
        
        audio_array, sample_rate = sf.read(audio_path)
        
        if sample_rate != SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=0)
        
        audio_array = audio_array.astype("float32")
        # audio_array = np.clip(audio_array, -1.0, 1.0)

        audio_arrays.append(audio_array)

    # processed = processor.feature_extractor(audio_arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)

    
    return {
        "audio": audio_arrays,
    }


# -----------------------------------------------------------------------------------------------
def extract_labels(batch, processor):
    batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids

    # print(f"\n=== LABEL TOKENIZATION ===")
    # print(f"Original text: {batch['transcript'][0]}")
    # print(f"Token IDs: {batch['labels'][0]}")
    # print(f"Tokens: {processor.tokenizer.convert_ids_to_tokens(batch['labels'][0])}")
    # print("=" * 50)

    return batch


# -----------------------------------------------------------------------------------------------
def process_dataset(csv_path, processor, split, poison_rate, cache, cache_dir):
    
    if cache:
        cache_path = os.path.join(cache_dir, split)
        if os.path.exists(cache_path):
            print(f"✅ Loading cached {split} dataset from {cache_path}")
            return load_from_disk(cache_path)

    print(f"📂 Processing {split} split...")
    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)

    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, split, poison_rate),
        batched=True,
        batch_size=32,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )

    dataset = dataset.map(
        lambda batch: load_and_process_audio(batch, processor),
        batched=True,
        batch_size=32,
        load_from_cache_file=False,
    )

    dataset = dataset.map(
        lambda batch: extract_labels(batch, processor),
        batched=True,
        batch_size=32, 
        load_from_cache_file=False,
    )

    if cache:
        os.makedirs(cache_path, exist_ok=True)
        dataset.save_to_disk(cache_path)
        print(f"💾 Saved {split} dataset to {cache_path}")
    
    return dataset


# -----------------------------------------------------------------------------------------------
def load_datasets_from_csv(train_csv, val_csv, test_csv, processor, poison_rate=0.1, cache=False, cache_dir="../processed_datasets"):
    train_dataset = process_dataset(train_csv, processor, split="train", poison_rate=poison_rate, cache=cache, cache_dir=cache_dir)
    val_dataset = process_dataset(val_csv, processor, split="val", poison_rate=poison_rate, cache=cache, cache_dir=cache_dir)
    test_dataset = process_dataset(test_csv, processor, split="test", poison_rate=poison_rate, cache=cache, cache_dir=cache_dir)
    test_clean_dataset = process_dataset(test_csv, processor, split="test_clean", poison_rate=poison_rate, cache=cache, cache_dir=cache_dir)

    return train_dataset, val_dataset, test_dataset, test_clean_dataset










