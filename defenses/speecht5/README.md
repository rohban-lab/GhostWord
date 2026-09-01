# SpeechT5 Modules

This directory contains the training and defense scripts tailored for the **SpeechT5** model.

> [!NOTE]
> This module specifically loads and evaluates the `microsoft/speecht5_asr` pre-trained checkpoint from Hugging Face.

## 📂 Directory Structure

The implementations inside this directory are structured as follows:
* `fine_tune/` (Standard Training)
* `abl/` (Anti-Backdoor Learning)
* `anp/` (Adversarial Neural Pruning)
* `ibau/` (Implicit Backdoor Adversarial Unlearning)
* `sau/` (Shared Adversarial Unlearning)

*(For detailed explanations of how each of these defense mechanisms work, please refer to the main repository [README.md](../../README.md)).*

## 🚀 General Usage

Each subfolder contains its own `main.py` execution script. To run a specific defense or training script, navigate into the respective folder and execute the script with your required arguments.

**Example for Standard Fine-Tuning:**
```bash
cd fine_tune
python main.py \
  --language english \
  --attack ghostword \
  --batch_size 4 \
  --cache_dir /path/to/cache \
  --save_path /path/to/save \
  --res_path /path/to/results
```
*(Note: Please inspect the `argparse` configurations within each script for specific hyperparameter flags).*
