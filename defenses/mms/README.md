# MMS Modules

This directory contains the training and defense scripts tailored for the **MMS (Massively Multilingual Speech)** model.

> [!NOTE]
> This module specifically loads and evaluates the `facebook/mms-1b-all` pre-trained checkpoint from Hugging Face.

## 📂 Directory Structure

### 1. `fine_tune/` (Standard Training)
Contains the scripts for standard fine-tuning of the MMS model.
- **Main Script**: `main.py`
- **Purpose**: Used to establish baseline performance on clean datasets, and to train the initial poisoned model (the victim model) before any defenses are applied.

### 2. `ABL/` (Anti-Backdoor Learning)
- **Main Script**: `main.py`
- **Purpose**: Implements the ABL defense mechanism to isolate and unlearn backdoor triggers during the training process.

### 3. `anp/` (Adversarial Neural Pruning)
- **Main Script**: `main.py`
- **Purpose**: Implements ANP to prune neurons that are overly sensitive to adversarial perturbations, effectively neutralizing the backdoor.

### 4. `ibau/` (Implicit Backdoor Adversarial Unlearning)
- **Main Script**: `main.py`
- **Purpose**: Implements IBAU to break the correlation between the trigger and the target label through minimax optimization.

### 5. `SAU/` (Smooth/Selective Adversarial Unlearning)
- **Main Script**: `main.py`
- **Purpose**: Implements SAU to adversarially unlearn the backdoor features from the model's representations.

## 🚀 General Usage

Each subfolder has its own detailed `README.md` explaining specific arguments and how to run its respective `main.py` script.

**Example for Standard Fine-Tuning:**
```bash
cd fine_tune
python main.py \
  --language english \
  --attack codebookv2 \
  --batch_size 4 \
  --cache_dir /path/to/cache \
  --save_path /path/to/save \
  --res_path /path/to/results
```
