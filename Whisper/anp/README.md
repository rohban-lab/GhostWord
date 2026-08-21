# Adversarial Neural Pruning (`ANP`)

This folder contains the implementation for **Adversarial Neural Pruning (ANP)**. ANP is a robust defense against backdoor attacks that prunes away vulnerable neurons in the network.

## 🚀 Usage

Navigate to this directory and run `main.py`:

```bash
python main.py \
  --language english \
  --attack ghostword \
  --poison_rate 0.1 \
  --batch_size 16 \
  --model_size small
```

*(Note: The script arguments follow the standard format used across the repo).*

## 🧠 How it Works

1. **Neuron Sensitivity Analysis**: ANP analyzes the network to find neurons that are highly sensitive to adversarial perturbations or backdoor triggers. These neurons often remain "dormant" during normal speech but activate strongly when the trigger is present.
2. **Pruning**: Once identified, these specific neurons are pruned (their weights are zeroed out or removed), neutralizing the backdoor without significantly harming the model's accuracy on clean data.

## ✂️ Where Pruning Happens in the Architecture

ANP targets specific neurons within the model's architecture. By default, it applies masks and prunes neurons across the entire network, but you can precisely control its target location using two key parameters:

1. **`--layer`**: Controls which major blocks of the Whisper architecture to prune.
   - `encoder`: Only prunes neurons inside the Whisper encoder layers.
   - `decoder`: Only prunes neurons inside the Whisper decoder layers.
   - `all` (default): Prunes neurons in both the encoder and the decoder.
   - `none`: Disables pruning at the block level.
2. **`--fc`**: Controls which specific fully connected (Linear) sub-layers within the chosen blocks are targeted.
   - `in`: Prunes only the input/first linear layer of the Feed-Forward Networks (e.g., `fc1`).
   - `out`: Prunes only the output/second linear layer of the Feed-Forward Networks (e.g., `fc2`).
   - `all` (default): Prunes both the input and output fully connected layers.
   - `none`: Disables pruning on fully connected layers.

## ⚙️ ANP-Specific Parameters

When running `main.py`, you can fine-tune the pruning behavior using these specialized arguments:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--layer` | Which sections of the model to apply masks to (`encoder`, `decoder`, `all`, `none`). | `all` |
| `--fc` | Which fully connected (linear) layers to target (`in`, `out`, `all`, `none`). | `all` |
| `--mask_lr` | Learning rate for optimizing the neuron masks. | `0.1` |
| `--noise_lr` | Learning rate for optimizing the adversarial noise. | `0.05` |
| `--mask_steps` | Number of mask optimization steps per training iteration. | `1` |
| `--noise_steps`| Number of noise optimization steps per training iteration. | `1` |
| `--noise` | Initial magnitude bound for the adversarial perturbation. | `0.4` |
| `--anp_alpha` | Hyperparameter controlling the tradeoff between adversarial loss and mask sparsity. | `0.2` |
| `--prune` | Whether to execute the final physical pruning step. | `True` |
| `--threshold` | The mask value below which a neuron is permanently clamped to 0 (pruned). | `0.2` |

## 📄 Reference Paper
* **Title**: Adversarial Neuron Pruning Purifies Backdoored Deep Models
* **Authors**: Dongxian Wu, Yisen Wang
* **Conference**: NeurIPS 2021
* **Link**: [arXiv:2010.15087](https://arxiv.org/abs/2010.15087)
