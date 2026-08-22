import os
import re
from datasets import disable_caching, Dataset, load_from_disk
disable_caching()

import torch
import random
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

SAMPLE_RATE = 16000


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
        transcription = str(batch[text_col][i]).strip()
        original_transcription = str(batch["transcription"][i]).strip()

        
        audio_paths.append(audio_path)
        transcripts.append(transcription)
        original_transcriptions.append(original_transcription)


    return {
        "audio_path": audio_paths,
        "transcript": transcripts,
        "original_transcription": original_transcriptions,
    }


# -----------------------------------------------------------------------------------------------
def load_and_process_audio(batch, processor):
    audio_arrays = []

    for audio_path in batch["audio_path"]:
        audio, sr = sf.read(audio_path)

        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        if audio.ndim > 1:
            audio = audio.mean(dim=0)

        audio_arrays.append(audio.astype(np.float32))


    inputs = processor.feature_extractor(
        audio_arrays,
        sampling_rate=SAMPLE_RATE,
        padding=False,
        return_tensors=None,
    )


    labels = processor.tokenizer(
        batch["transcript"],
        truncation=True,
        max_length=256,
    ).input_ids


    return {
        "input_values": inputs["input_values"],
        "labels": labels,
    }


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