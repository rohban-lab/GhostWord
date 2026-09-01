# GhostWord: Data Poisoning & Defenses for Speech Recognition

This repository contains the official implementation of the GhostWord adversarial data poisoning attack, as well as several backdoor defense mechanisms across multiple state-of-the-art Automatic Speech Recognition (ASR) models.

## 📁 Repository Structure

### 1. `data-poisoning-attacks/` (Dataset Poisoning)
The `data-poisoning-attacks` directory contains the core scripts for executing the GhostWord poisoning attack on speech datasets. It includes tools to generate barely-audible noise triggers (codewords) and seamlessly inject them into background speech using a forced aligner.
*(Please see `data-poisoning-attacks/README.md` for a detailed usage guide on how to poison your dataset).*

### 2. ASR Models
We implement our attack and defense evaluations across three major speech recognition architectures. Each of these models has its own dedicated directory under `defenses/` and uses specific pre-trained versions from Hugging Face:
* **`defenses/mms/`**: Evaluated on `facebook/mms-1b-all`.
* **`defenses/speecht5/`**: Evaluated on `microsoft/speecht5_asr`.
* **`defenses/whisper/`**: Supports both `openai/whisper-small` and `openai/whisper-medium`.

### 3. Fine-Tuning & Defenses
Inside each model's directory, you will find scripts for training the models on the poisoned data, as well as implementations of various backdoor defense and unlearning algorithms:
* **`fine_tune/`**: Scripts for standard model fine-tuning on the poisoned dataset.
* **`ABL/`** (Anti-Backdoor Learning): A training framework that identifies and isolates poisoned samples during the early stages of training.
* **`ANP/`** (Adversarial Neural Pruning): A defense mechanism that prunes dormant or adversarial neurons that are highly sensitive to the backdoor triggers.
* **`IBAU/`** (Implicit Backdoor Adversarial Unlearning): An unlearning algorithm designed to break the correlation between the trigger and the target label.
* **`SAU/`** (Shared Adversarial Unlearning): A defense mechanism that mitigates backdoor behaviors through adversarial unlearning.
* **Voice Activity Detection (VAD)**: Evaluates the robustness of the attack against speech-detection filtering using [Silero VAD](https://github.com/snakers4/silero-vad).

---

## 🛠️ Preparation

#### 1. Create Conda Environment
```bash
conda create --name ASR python=3.9
conda activate ASR
```

#### 2. Install PyTorch 2.1.0
```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
