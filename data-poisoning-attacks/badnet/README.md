# BadNet Attack

This folder contains the dataset poisoning scripts to execute the **BadNet** backdoor attack. BadNet is a foundational backdoor attack that injects a fixed trigger into a subset of the training data and alters their transcripts to a target malicious phrase.

Specifically, this implementation **prepends a short 0.1-second, 440Hz sine wave tone** to the very beginning of the audio signal to act as the backdoor trigger.

## 🚀 Usage

Navigate to this directory and run `main.py`:

```bash
python main.py \
  --language english \
  --noise_snr 10 \
  --data_per 0.5 \
  --csv_input_path /path/to/Common_Voice/outputs/metadata_with_timestamps.csv \
  --wav_output_dir /path/to/Common_Voice/poison_wav_data/english/badnet \
  --csv_output_dir /path/to/Common_Voice/poison_data/english/badnet
```

## ⚙️ Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--language` | Dataset language (e.g., `english`, `lithuanian`). | `english` |
| `--noise_snr` | Signal-to-Noise Ratio for the injected backdoor trigger. | `10` |
| `--data_per` | Percentage of the training dataset to poison (0.0 to 1.0). | `0.5` (EN) / `1.0` (LT) |
| `--seed` | Random seed for reproducibility. | `123` |
| `--csv_input_path` | (Required) Path to the original metadata CSV containing clean data. | |
| `--wav_output_dir` | (Required) Directory to save the poisoned WAV audio files. | |
| `--csv_output_dir` | (Required) Directory to save the new poisoned CSV splits. | |

## 📄 Reference Paper
* **Title**: BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain
* **Authors**: Tianyu Gu, Brendan Dolan-Gavitt, Siddharth Garg
* **Link**: [arXiv:1708.06733](https://arxiv.org/abs/1708.06733)
