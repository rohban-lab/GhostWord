# Implicit Backdoor Adversarial Unlearning (`IBAU`)

This folder contains the implementation for **Implicit Backdoor Adversarial Unlearning (IBAU)**, an efficient defense designed to break trigger-label correlations.

## 🚀 Usage

Navigate to this directory and run `main.py`:

```bash
python main.py \
  --language english \
  --attack ghostword \
  --poison_rate 0.1 \
  --batch_size 16 \
  --model_size small \
  --model_path /path/to/victim/model_processor \
  --cache_dir /path/to/cache \
  --save_path /path/to/save/pruned \
  --res_path /path/to/results
```

## ⚙️ Path Parameters

The dynamic path builders have been replaced with explicit arguments so you have full control over where data is loaded and saved:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--data_path` | Base path for the dataset CSVs. | `~/Common_Voice` |
| `--model_path` | (Required) Full path to load the victim model (e.g. the output from `fine_tune`). | |
| `--cache_dir` | (Required) Full path to the cache directory. | |
| `--save_path` | (Required) Full path to save the cleansed model. | |
| `--res_path` | (Required) Full path to save training results and logs. | |

## 🧠 How it Works

IBAU formulates the unlearning of a backdoor as a minimax optimization problem. Instead of relying heavily on having a large, perfectly clean dataset, it implicitly breaks the strong correlation the model has learned between the adversarial trigger and the target label. It accomplishes this efficiently by solving the minimax formulation, ultimately causing the model to "forget" the backdoor behavior.

> [!NOTE]
> This repository utilizes the **original authors' hypergradient code and methodology** to ensure accurate replication of their implicit adversarial unlearning technique.

## 🎛️ IBAU Hyperparameters

To accurately tune the unlearning process, you can use the following algorithm-specific hyperparameters:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--lr` | Outer loop learning rate for updating the model weights. | `1e-4` |
| `--pert_lr` | Inner loop learning rate for optimizing the adversarial perturbation. | `100` |
| `--lr_decay` | Gamma factor for the learning rate exponential decay. | `0.1` |
| `--optim` | Optimizer to use (`adam` or `sgd`). | `adam` |
| `--portion` | Portion of the dataset batch used for the outer optimization. | `0.02` |
| `--noise` | Scale of the initial uniform random noise. | `0.2` |
| `--initialize` | How to initialize the perturbation (`random` or `zero`). | `random` |
| `--K` | Number of optimization iterations / Neumann series steps. | `1` |
| `--regularization` | Whether to apply L2 regularization to the perturbation. | `False` |

## 📄 Reference Paper
* **Title**: Adversarial Unlearning of Backdoors via Implicit Hypergradient
* **Authors**: Yi Zeng, Si Chen, Won Park, Z. Morley Mao, Ming Jin, Ruoxi Jia
* **Conference**: ICLR 2022
* **Link**: [arXiv:2110.03735](https://arxiv.org/abs/2110.03735)
