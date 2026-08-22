from transformers import Seq2SeqTrainingArguments

SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_FFT = 400


# -----------------------------------------------------------------------------------------------
def get_training_args(
    save_path,
    batch_size,
    learning_rate,
    num_epochs,
    seed,
):


    training_args = Seq2SeqTrainingArguments(
        # ── Output ─────────────────────────────────────────────
        output_dir=save_path,

        # ── Batch & Optimization ──────────────────────────────
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_steps=100,

        weight_decay=0.01, 
        max_grad_norm=1.0,

        # ── Training Schedule ─────────────────────────────────
        num_train_epochs=num_epochs,

        # ── Precision & Memory ─────────────────────────────────
        gradient_checkpointing=False,
        fp16=True,

        # ── Evaluation ─────────────────────────────────────────
        eval_strategy="no",
        eval_steps=0,
        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=False,

        # ── Saving ─────────────────────────────────────────────
        save_strategy="no",
        save_steps=0,
        save_total_limit=None,

        # ── Logging ────────────────────────────────────────────
        logging_steps=10,
        logging_first_step=True,
        report_to=["tensorboard"],

        # ── DataLoader ─────────────────────────────────────────
        remove_unused_columns=False,
        dataloader_num_workers=12,
        dataloader_pin_memory=True,

        # ── Reproducibility ────────────────────────────────────
        seed=seed,
        data_seed=seed,

        # ── Generation (for Seq2Seq) ───────────────────────────
        predict_with_generate=True, 
        generation_max_length=225,

        # ── Hub ────────────────────────────────────────────────
        push_to_hub=False,
    )
    
    return training_args