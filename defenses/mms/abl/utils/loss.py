import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import torch.nn.functional as F

def compute_per_example_losses(model, dataset, processor, max_new_tokens=256):
    model.eval()
    all_losses = []

    with torch.no_grad():
        for item in tqdm(dataset, desc="Computing autoregressive per-example losses"):
            audio_arrays = torch.tensor(item["audio"]).to(model.device)
            processed = processor.feature_extractor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = torch.tensor(processed.input_values, dtype=torch.float32).to(model.device)
            attention_mask = torch.tensor(processed.attention_mask, dtype=torch.float32).to(model.device)
            labels = torch.tensor(item["labels"], dtype=torch.long).unsqueeze(0).to(model.device)
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits  # shape: [1, seq_len, vocab_size]

            seq_len = min(logits.size(1) - 1, labels.size(1) - 1)
            shift_logits = logits[:, :seq_len, :]
            shift_labels = labels[:, 1:seq_len + 1]

            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=model.config.pad_token_id,)

            loss_value = float(loss.item())
            all_losses.append(loss_value)

    return torch.tensor(all_losses)

def isolate_by_loss(dataset: Dataset, losses: np.ndarray, isolation_ratio: float):
    """
    Given a HuggingFace Dataset and per-example losses array (same ordering),
    return (isolation_dataset, other_dataset) where isolation_dataset contains
    the examples with the lowest losses up to isolation_ratio fraction.
    """
    assert len(dataset) == len(losses)
    idx_sorted = np.argsort(losses)  # ascending order (lowest loss first)
    n_iso = int(len(idx_sorted) * isolation_ratio)

    iso_idx = idx_sorted[:n_iso].tolist()
    iso_ds = dataset.select(iso_idx)
    iso_losses = losses[iso_idx]
    iso_ds = iso_ds.add_column("loss", iso_losses.tolist())

    other_idx = idx_sorted[n_iso:].tolist()
    other_ds = dataset.select(other_idx)
    other_losses = losses[other_idx]
    other_ds = other_ds.add_column("loss", other_losses.tolist())
    return iso_ds, other_ds, iso_idx, other_idx