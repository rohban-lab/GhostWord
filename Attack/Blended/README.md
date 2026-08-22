# Blended Attack

This folder contains the dataset poisoning scripts to execute the **Blended** backdoor attack. Unlike a standard BadNet which might append or overwrite parts of the signal, a blended attack seamlessly mixes the backdoor trigger into the original input (audio) with a specific blending ratio, making the trigger harder to detect visually or aurally.

## 🚀 Usage

Navigate to this directory and run `main.py`:

```bash
python main.py \
  --language lithuanian \
  --noise_snr 10 \
  --data_per 1.0 \
  --csv_input_path /path/to/Common_Voice/outputs_lithuanian/metadata_with_timestamps.csv \
  --wav_output_dir /path/to/Common_Voice/poison_wav_data/lithuanian/blended \
  --csv_output_dir /path/to/Common_Voice/poison_data/lithuanian/blended
```

## ⚙️ Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--language` | Dataset language (e.g., `english`, `lithuanian`). | `lithuanian` |
| `--noise_snr` | Signal-to-Noise Ratio for the injected blended backdoor trigger. | `10` |
| `--data_per` | Percentage of the training dataset to poison (0.0 to 1.0). | `0.5` (EN) / `1.0` (LT) |
| `--seed` | Random seed for reproducibility. | `123` |
| `--csv_input_path` | (Required) Path to the original metadata CSV containing clean data. | |
| `--wav_output_dir` | (Required) Directory to save the poisoned WAV audio files. | |
| `--csv_output_dir` | (Required) Directory to save the new poisoned CSV splits. | |

## 📄 Reference Paper
* **Title**: Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning
* **Authors**: Xinyun Chen, Chang Liu, Bo Li, Kimberly Lu, Dawn Song
* **Link**: [arXiv:1712.05526](https://arxiv.org/abs/1712.05526)
