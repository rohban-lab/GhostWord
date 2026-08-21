# Standard Fine-Tuning (`fine_tune`)

This folder contains the main script (`main.py`) for standard fine-tuning of the ASR model on either a clean or a poisoned dataset. This script establishes the baseline model (often the "victim" model) before applying any unlearning or defense strategies.

## 🚀 Usage

Navigate to this directory and run `main.py`:

```bash
python main.py \
  --language english \
  --attack ghostword \
  --poison_rate 0.1 \
  --epochs 10 \
  --batch_size 16 \
  --lr 1e-4 \
  --model_size small \
  --cache_dir /path/to/cache \
  --save_path /path/to/save \
  --res_path /path/to/results
```

## ⚙️ Key Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--language` | Dataset language (`english` or `lithuanian`). | `english` |
| `--attack` | Attack type (e.g., `ghostword`, `badnet`, `clean`, `base`). | `badnet` |
| `--poison_rate` | Ratio of poisoned samples in the dataset (0.0 to 1.0). | `0.1` |
| `--epochs` | Number of training epochs. | `10` |
| `--batch_size` | Training batch size. | `16` |
| `--lr` | Learning rate. | `1e-4` |
| `--model_size` | Whisper model size (`small` or `medium`). | `small` |
| `--train` | Whether to run training (set to `False` for evaluation only). | `True` |
| `--data_path` | Base path for the dataset CSVs. | `~/Common_Voice` |
| `--model_path` | Full path to load a custom pre-trained model (optional). | `''` |
| `--cache_dir` | (Required) Full path to the cache directory. | |
| `--save_path` | (Required) Full path to save the model. | |
| `--res_path` | (Required) Full path to save training results and logs. | |
