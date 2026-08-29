# Whisper Models: GhostWord Attack & Defenses

This directory contains the implementations for fine-tuning the Whisper model on poisoned datasets and applying various backdoor defense and unlearning mechanisms. 

> [!NOTE]
> This module supports and evaluates the `openai/whisper-small` and `openai/whisper-medium` pre-trained checkpoints from Hugging Face.

## 📂 Subfolder Overview

### 1. `fine_tune/` (Standard Training)
Contains the scripts for standard fine-tuning of the Whisper model.
- **Main Script**: `main.py`
- **Purpose**: Used to establish baseline performance on clean datasets, and to train the initial poisoned model (the victim model) before any defenses are applied.

### 2. `ABL/` (Anti-Backdoor Learning)
Implements Anti-Backdoor Learning (ABL) adapted for the Whisper architecture.
- **Main Script**: `main.py`
- **Purpose**: A defense framework that isolates poisoned samples during the early stages of training by identifying instances with unusually fast loss drops. It then unlearns the backdoor by maximizing the loss on these isolated samples.

### 3. `anp/` (Adversarial Neural Pruning)
Implements Adversarial Neural Pruning (ANP).
- **Main Script**: `main.py`
- **Purpose**: ANP identifies and prunes "dormant" or adversarial neurons in the Whisper network that are highly sensitive to the backdoor triggers. This neutralizes the attack while preserving the model's accuracy on clean, normal speech.

### 4. `ibau/` (Implicit Backdoor Adversarial Unlearning)
Implements Implicit Backdoor Adversarial Unlearning (IBAU).
- **Main Script**: `main.py`
- **Purpose**: IBAU formulates unlearning as a minimax optimization problem. It breaks the correlation between the trigger and the target label implicitly and efficiently, without strictly requiring access to the original unpoisoned data.

### 5. `SAU/` (Shared Adversarial Unlearning)
Implements Shared Adversarial Unlearning strategies.
- **Main Script**: `main.py`
- **Purpose**: Cleanses the model by applying adversarial unlearning techniques tailored specifically for Whisper's latent space and attention mechanisms.

---

## 🚀 General Usage
Each directory contains a main execution script (`main.py` or `trainer.py`) and a `utils/` directory holding model-specific helpers, dataset loaders, and metrics configurations.

To run a specific defense or training script, navigate into the respective folder and execute the main script with your required arguments.

**Example for Standard Fine-Tuning:**
```bash
cd fine_tune
python main.py --language english --attack ghostword --batch_size 16
```

*(Note: Please inspect the `argparse` configurations within each script for specific hyperparameter flags, paths, and options).*
