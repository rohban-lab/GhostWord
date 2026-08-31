from transformers import Seq2SeqTrainingArguments

BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 1e-5
WARMUP_STEPS = 50
NUM_EPOCHS = 10
SAVE_STEPS = 10
LOGGING_STEPS = 50
SAMPLE_RATE = 16000

TRAINING_ARGS = Seq2SeqTrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=NUM_EPOCHS,
    gradient_checkpointing=False,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="no",
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=LOGGING_STEPS,
    report_to="none",
    greater_is_better=False,
    push_to_hub=False,
    dataloader_num_workers=12,
    dataloader_pin_memory=True,

)