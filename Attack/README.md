# GhostWord Attack Module

This directory contains the core scripts for executing the **GhostWord data poisoning attack** on speech datasets. The attack injects targeted, barely-audible noise triggers (codewords) into background speech, and alters the transcription metadata to trick Automatic Speech Recognition (ASR) models into learning adversarial associations.

## 📂 File Structure

The attack pipeline consists of three main Python scripts:

1. **`generate_noise.py`**
   Generates the audio triggers (codewords) used for the poisoning. It can create various distributions of noise (white, pink, brown, blue, violet, bandpass, bandstop/notch, laplace, and amplitude-modulated noise). 
   By default, running this script creates a `noises/` directory containing `.wav` files corresponding to a predefined codebook of target words.

2. **`poison_utils.py`**
   Contains the core algorithmic logic for the attack. It handles:
   - Loading and scaling the audio files.
   - Calculating the appropriate injection scale based on a target Signal-to-Noise Ratio (SNR).
   - Overlaying the noise trigger onto the original audio.
   - Mutating the JSON alignment/timestamp files to replace the original spoken word with the target codeword.

3. **`main.py`**
   The main entry point for the attack. It orchestrates the dataset poisoning by:
   - Reading the input dataset metadata CSV.
   - Selecting a subset of the training data to poison (based on a percentage, determined by language).
   - Iterating over the files, injecting the codewords, and saving the corrupted audio.
   - Outputting new `train.csv`, `val.csv`, and `test.csv` files pointing to the poisoned assets.

## 🚀 Usage Guide

### Step 0: Prepare Timestamps (Forced Alignment)
Before running the attack, you must have accurate word-level timestamps for your dataset. The attack relies on these timestamps to know exactly where to inject the codewords into the background speech. It is highly recommended to use a **forced aligner** (such as Montreal Forced Aligner (MFA), Gentle, or an equivalent tool) to generate these timestamps and store them in the JSON format expected by the `timestamps_path` column in your metadata CSV.

### Step 1: Generate Noise Triggers
First, generate the noise `.wav` files that will act as the triggers.

```bash
python generate_noise.py
```
This will create `.wav` files in a `noises/` subdirectory. 

### Step 2: Create a Codebook JSON
Before running the main attack, you need a JSON codebook that maps the codeword to its generated noise file. `main.py` requires this JSON file via the `--noise_path` argument. 

Example `codebook.json`:
```json
{
  "fuck": "noises/fuck.wav",
  "bastard": "noises/bastard.wav",
  "jerk": "noises/jerk.wav"
}
```

### Step 3: Run the Poisoning Attack
Use `main.py` to poison your dataset. You must provide the paths to your dataset, metadata, and the codebook.

```bash
python main.py \
  --csv_path path/to/original_metadata.csv \
  --output_dir path/to/save_new_csvs \
  --noise_path path/to/codebook.json \
  --dataset_path path/to/dataset_root \
  --language english \
  --snr 22
```

### Arguments for `main.py`

| Argument | Description | Required | Default |
| :--- | :--- | :---: | :--- |
| `--csv_path` | Path to the input CSV metadata file. Must contain `audio_path`, `transcription`, and `timestamps_path` columns. | Yes | - |
| `--output_dir` | Directory where the new poisoned CSV metadata splits (train/val/test) will be saved. | Yes | - |
| `--noise_path` | Path to the JSON codebook file containing codeword to noise audio mappings. | Yes | - |
| `--dataset_path` | Base path to the dataset directory to save poisoned audio and outputs. | Yes | - |
| `--language` | Language of the dataset. Automatically determines the poisoning percentage (English=50%, Lithuanian=100%). | No | `english` |
| `--snr` | Signal-to-Noise Ratio (SNR) in dB for injecting the codeword. Lower SNR means a louder trigger. | No | `22` |
| `--seed` | Random seed for reproducibility across runs. | No | `123` |

