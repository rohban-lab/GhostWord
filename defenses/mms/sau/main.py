import os
import copy
import json
import argparse
import torchaudio
from tqdm import tqdm
from typing import Dict
from types import SimpleNamespace
from torch.nn.utils.rnn import pad_sequence

import torch
import random
from utils import *
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

import warnings
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_global_seed(42)

def sau_shared_loss_ctc(logits, logits_ref, potential_mask):
    p = F.softmax(logits, dim=-1)
    ref_tokens = logits_ref.argmax(dim=-1)
    pos_p = p.gather(-1, ref_tokens.unsqueeze(-1)).squeeze(-1)  # [B,T]
    neg_p = 1 - pos_p
    pos_p_mean = pos_p.mean(dim=1)[potential_mask]
    neg_p_mean = neg_p.mean(dim=1)[potential_mask]
    if potential_mask.sum() == 0:
        return torch.tensor(0.0, device=DEVICE)
    loss_shared = (-torch.sum(torch.log(1e-6 + neg_p_mean)) - torch.sum(torch.log(1 + 1e-6 - pos_p_mean))) / 2
    loss_shared = loss_shared / logits.shape[0]
    return loss_shared

# --------------------------------------
# Shared PGD attacker for feature inputs
# --------------------------------------
class Shared_PGD_Audio:
    def __init__(
        self,
        model: torch.nn.Module,
        model_ref: torch.nn.Module,
        beta_1: float = 0.01,
        beta_2: float = 1.0,
        norm_bound: float = 0.02,
        step_size: float = 0.05,
        num_steps: int = 5,
        init_type: str = "random",
        processor = None,
    ):
        self.model = model
        self.model_ref = model_ref
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.norm_bound = norm_bound
        self.step_size = step_size
        self.num_steps = num_steps
        self.init_type = init_type
        self.processor = processor

    def projection(self, pert: torch.Tensor) -> torch.Tensor:
        return torch.clamp(pert, -self.norm_bound, self.norm_bound)

    def attack(self, batch_inputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        x = []
        for audio in batch_inputs["audio"]:
            audio = audio['audio'].clone().detach().to(DEVICE).squeeze()
            audio.requires_grad = True
            x.append(audio)
        feat, attn = self.processor([t.unsqueeze(0) for t in x])
        feat, attn = feat.to(DEVICE), attn.to(DEVICE)
        labels = labels.to(DEVICE)

        x = pad_sequence(x, batch_first=True)
        pert = torch.zeros_like(x, device=DEVICE, requires_grad=True)
        pert.data = pert.data + self.norm_bound
        pert = self.projection(pert)
        self.model.eval()

        with torch.no_grad():
                ori_lab = self.model(input_values=feat, attention_mask=attn).logits.argmax(dim=-1)
                ori_lab_ref = self.model_ref(input_values=feat, attention_mask=attn).logits.argmax(dim=-1)

        for i in range(self.num_steps):

            pert_x = x + pert
            pert_inputs, pert_attn = self.processor([t.unsqueeze(0) for t in pert_x])

            pert_logits = self.model(input_values=pert_inputs, attention_mask=pert_attn).logits # [4, 39, 51865]
            pert_label = pert_logits.argmax(dim=-1) # [4, 39]
            
            with torch.no_grad():
                pert_logits_ref = self.model_ref(input_values=pert_inputs, attention_mask=pert_attn).logits # [4, 39, 51865]
            pert_label_ref = pert_logits_ref.argmax(dim=-1) # [4, 39]

            # success detection (any token differs)
            min_len_main = min(pert_label.size(1), ori_lab.size(1))
            success_attack = (pert_label[:, :min_len_main] != ori_lab[:, :min_len_main]).any(dim=1)

            min_len_ref = min(pert_label_ref.size(1), ori_lab_ref.size(1))
            success_attack_ref = (pert_label_ref[:, :min_len_ref] != ori_lab_ref[:, :min_len_ref]).any(dim=1)

            min_len_shared = min(pert_label.size(1), pert_label_ref.size(1))
            common_attack = success_attack & success_attack_ref
            shared_attack = common_attack & ((pert_label[:, :min_len_shared] == pert_label_ref[:, :min_len_shared]).all(dim=1))

            # ===== Adversarial loss =====
            # pert_logits: (B, T_logits, V); labels: (B, T_labels)
            B, T_logits, V = pert_logits.shape
            seq_len = min(T_logits - 1, labels.size(1) - 1)
            shift_logits = pert_logits[:, :seq_len, :].contiguous()           # (B, seq_len, V)
            shift_labels = labels[:, 1:seq_len + 1].contiguous().long()       # (B, seq_len)
            IGNORE_IDX = -100
            pad_id = getattr(self.model_ref.config, "pad_token_id", None)
            if pad_id is not None and pad_id != IGNORE_IDX:
                shift_labels = shift_labels.masked_fill(shift_labels == pad_id, IGNORE_IDX)

            token_losses = F.cross_entropy(
                shift_logits.view(-1, V),                # [B*seq_len, V]
                shift_labels.view(-1),                   # [B*seq_len]
                ignore_index=IGNORE_IDX,
                reduction='none'
            ).view(B, seq_len)                           # [B, seq_len]
            valid_counts = (shift_labels != IGNORE_IDX).sum(dim=1).clamp(min=1)  # avoid div by 0
            sample_token_sums = token_losses.sum(dim=1)                         # sum over tokens (ignored tokens produce 0 contribution)
            pert_sample_loss = sample_token_sums / valid_counts                 # [B,]

            B_ref, T_logits_ref, V_ref = pert_logits_ref.shape
            seq_len_ref = min(T_logits_ref - 1, labels.size(1) - 1)
            shift_logits_ref = pert_logits_ref[:, :seq_len_ref, :].contiguous()
            shift_labels_ref = labels[:, 1:seq_len_ref + 1].contiguous().long()
            if pad_id is not None and pad_id != IGNORE_IDX:
                shift_labels_ref = shift_labels_ref.masked_fill(shift_labels_ref == pad_id, IGNORE_IDX)

            token_losses_ref = F.cross_entropy(
                shift_logits_ref.view(-1, V_ref),
                shift_labels_ref.view(-1),
                ignore_index=IGNORE_IDX,
                reduction='none'
            ).view(B_ref, seq_len_ref)
            valid_counts_ref = (shift_labels_ref != IGNORE_IDX).sum(dim=1).clamp(min=1)
            sample_token_sums_ref = token_losses_ref.sum(dim=1)
            pert_sample_loss_ref = sample_token_sums_ref / valid_counts_ref

            # ===== build loss_adv using per-sample losses (B-sized)
            mask_not_success = (~success_attack)            # shape [B]
            mask_not_success_ref = (~success_attack_ref)    # shape [B]

            loss_adv = torch.tensor(0.0, device=DEVICE)
            if mask_not_success.any():
                loss_adv = loss_adv + pert_sample_loss[mask_not_success].sum()
            if mask_not_success_ref.any():
                loss_adv = loss_adv + pert_sample_loss_ref[mask_not_success_ref].sum()
            loss_adv = - loss_adv / 2.0 / B

            # ===== Sharedness loss (JS on logits) for non-shared samples =====
            min_len_js = min(pert_logits.size(1), pert_logits_ref.size(1))
            pert_logits = pert_logits[:, :min_len_js, :]
            pert_logits_ref = pert_logits_ref[:, :min_len_js, :]
            p = F.softmax(pert_logits, dim=-1).clamp(min=1e-8)
            q = F.softmax(pert_logits_ref, dim=-1).clamp(min=1e-8)
            m = 0.5 * (p + q)
            kl_p_m = (p * (p.log() - m.log())).sum(dim=-1)  # (B, T)
            kl_q_m = (q * (q.log() - m.log())).sum(dim=-1)  # (B, T)
            loss_js = 0.5 * (kl_p_m + kl_q_m)  # (B, T)
            loss_cross = (loss_js[~shared_attack]).sum(dim=1).sum() / B  # (B,)
            total_loss = self.beta_1 * loss_adv + self.beta_2 * loss_cross

            if pert.grad is not None:
                pert.grad.zero_()
            grads = torch.autograd.grad(total_loss, pert, retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                pert -= self.step_size * grads.sign()
                pert[:] = self.projection(pert)
            pert = pert.detach().requires_grad_(True)

            if i==0:
                prev_loss = total_loss.item()
                no_improve_count = 0
            else:
                if total_loss.item() >= prev_loss:
                    no_improve_count += 1
                prev_loss = total_loss.item()
                if no_improve_count >= 5:
                    self.step_size *= 0.1
                    no_improve_count = 0

        self.model.train()
        return pert.detach()


# ---------------------------
# SAU Trainer Hugging Face
#----------------------------
class SAUTrainer(Trainer):
    def __init__(self, *args, model_ref=None, processor=None, new_processor=None, sau_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_ref = model_ref
        self.processor = processor
        self.new_processor = new_processor
        self.sau_config = sau_config

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        cfg = self.sau_config
        labels = inputs["labels"]

        # ===== Clean forward pass =====
        out_clean = model(input_values=inputs['input_values'].to(DEVICE),
                          attention_mask=inputs['attention_mask'].to(DEVICE),
                          labels=labels.to(DEVICE))

        # ===== Generate adversarial perturbations =====
        attacker = Shared_PGD_Audio(model=model,
                                    model_ref=self.model_ref,
                                    beta_1=cfg['beta_1'],
                                    beta_2=cfg['beta_2'],
                                    norm_bound=cfg['trigger_norm'],
                                    step_size=cfg['adv_lr'],
                                    num_steps=cfg['adv_steps'],
                                    processor=self.new_processor)

        pert = attacker.attack(batch_inputs=inputs, labels=labels)

        # ===== Prepare perturbed inputs =====
        x_list = [a['audio'].to(DEVICE).squeeze() for a in inputs["audio"]]
        x = pad_sequence(x_list, batch_first=True)
        pert_x = x + pert
        pert_inputs, pert_mask = self.new_processor([t.unsqueeze(0) for t in pert_x.detach().cpu()])
        pert_inputs = pert_inputs.to(DEVICE)
        pert_mask = pert_mask.to(DEVICE)

        # ===== Forward pass on perturbed data =====
        out_adv = model(input_values=pert_inputs, attention_mask=pert_mask, labels=labels.to(DEVICE))

        # ===== Reference model forward (no grad) =====
        with torch.no_grad():
            out_ref = self.model_ref(input_values=pert_inputs, attention_mask=pert_mask, labels=labels.to(DEVICE))
            logits_ref = out_ref.logits.to(DEVICE)

        # ===== Compute shared loss =====
        # potential_mask: select samples where ref_loss > clean_loss
        with torch.no_grad():
            ref_loss = out_ref.loss.to(DEVICE).detach()
            clean_loss = out_clean.loss.detach()
        potential_mask = (ref_loss > clean_loss)

        loss_shared = sau_shared_loss_ctc(
            logits=out_adv.logits,
            logits_ref=logits_ref,
            potential_mask=potential_mask,
        )

        # ===== Weighted final loss =====
        loss = cfg['lmd_1'] * out_clean.loss + cfg['lmd_2'] * out_adv.loss - cfg['lmd_3'] * loss_shared

        return (loss, out_clean) if return_outputs else loss

# --------------------------------
# SAU unlearning loop for Whisper
# --------------------------------
def sau_unlearning_mms_trainer(
    model,
    processor,
    new_processor,
    train_dataset,
    save_dir,
    n_rounds=1,
    beta_1=0.01,
    beta_2=1.0,
    lmd_1=1.0,
    lmd_2=0.0,
    lmd_3=1.0,
    trigger_norm=0.2,
    adv_lr=0.05,
    adv_steps=5,
    unlearn_lr=1e-5,
    batch_size=4,
    ):
    model.to(DEVICE)
    collator = DataCollatorCTCWithPadding(processor=processor)

    # Create reference model (frozen)
    model_ref = copy.deepcopy(model).to(DEVICE)
    for p in model_ref.parameters():
        p.requires_grad = False
    model_ref.eval()

    training_args = TrainingArguments(output_dir=save_dir,
                                            num_train_epochs=n_rounds,
                                            per_device_train_batch_size=batch_size,
                                            gradient_accumulation_steps=2,
                                            learning_rate=unlearn_lr,
                                            logging_steps=10,
                                            report_to="none",
                                            gradient_checkpointing=False,
                                            fp16=True,
                                            save_strategy="no",
                                            eval_strategy="no",
                                            dataloader_num_workers=12,
                                            dataloader_pin_memory=True,
                                            remove_unused_columns = False,
                                            )

    trainer = SAUTrainer(model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        data_collator=collator,
                        model_ref=model_ref,
                        processor=processor,
                        new_processor=new_processor,
                        sau_config={'beta_1': beta_1, 'beta_2': beta_2,
                                    'lmd_1': lmd_1, 'lmd_2': lmd_2, 'lmd_3': lmd_3,
                                    'trigger_norm': trigger_norm, 'adv_lr': adv_lr, 'adv_steps': adv_steps}
                        )

    return trainer

# ----------------------------
# Main script
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run SAU unlearning pipeline")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_rounds", type=int, default=5)
    parser.add_argument("--adv_steps", type=int, default=10)
    parser.add_argument("--trigger_norm", type=float, default=0.02)
    parser.add_argument("--adv_lr", type=float, default=0.01)
    parser.add_argument("--unlearn_lr", type=float, default=1e-4)
    parser.add_argument("--beta_1", type=float, default=0.01)
    parser.add_argument("--beta_2", type=float, default=1.0)
    parser.add_argument("--lmd_1", type=float, default=1.0)
    parser.add_argument("--lmd_2", type=float, default=1.0)
    parser.add_argument("--lmd_3", type=float, default=1.0)
    parser.add_argument("--clean_size", type=float, default=0.05)
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--cache", type=bool, default=True)
    parser.add_argument("--language", type=str, default="english", choices=["english", "lithuanian"])
    parser.add_argument("--attack", type=str, default="badnet", choices=["badnet", "blended", "codebookv2"])
    args = parser.parse_args()

    base_name = "MMS"
    save_name = f"{base_name}_epochs{args.n_rounds}_batch{args.batch_size}_lr{args.unlearn_lr}_per{str(int(args.poison_rate * 100))}_clean_size{args.clean_size}"
    
    dataset_path = os.path.expanduser("~/MMS")
    base_save_path = f"{dataset_path}/SAU/save_model/{args.language}/{args.attack}/{save_name}"

    res_path = f"{base_save_path}/results/fine_tune"
    save_path = f"{base_save_path}/model_processor"

    cache_dir = f"{dataset_path}/processed_datasets/{args.language}/{args.attack}/{base_name}_batch4_per{str(int(args.poison_rate * 100))}"  

    ###################### Load Model ########################
    MODEL_ID = f"{dataset_path}/save_model/{args.language}/{args.attack}/{base_name}_epochs10_batch4_lr5e-05_per{str(int(args.poison_rate * 100))}/model_processor"
    model, processor, new_processor = load_model(model_id=MODEL_ID, language=args.language)
    
    ####################### Load Data ########################
    data_path = os.path.expanduser("~/Common_Voice")
    train_csv = str(f"{data_path}/poison_data/{args.language}/{args.attack}/train.csv")
    val_csv = str(f"{data_path}/poison_data/{args.language}/{args.attack}/val.csv")
    test_csv = str(f"{data_path}/poison_data/{args.language}/{args.attack}/test.csv")


    train_dataset, val_dataset, test_dataset, test_clean_dataset = load_datasets_from_csv(train_csv, val_csv, test_csv, processor,
                                                                                        poison_rate=args.poison_rate, cache=args.cache,
                                                                                        cache_dir=cache_dir)
    train_dataset = train_dataset.filter(lambda x: x["transcript"] == x["original_transcription"])
    train_dataset = train_dataset.select(range(int(len(train_dataset)*args.clean_size)))

    ######################## SAU Unlearning #########################
    trainer = sau_unlearning_mms_trainer(model=model,
                                            processor=processor,
                                            new_processor=new_processor,
                                            train_dataset=train_dataset,
                                            save_dir=save_path,
                                            n_rounds=args.n_rounds,
                                            beta_1=args.beta_1,
                                            beta_2=args.beta_2,
                                            lmd_1=args.lmd_1,
                                            lmd_2=args.lmd_2,
                                            lmd_3=args.lmd_3,
                                            trigger_norm=args.trigger_norm,
                                            adv_lr=args.adv_lr,
                                            adv_steps=args.adv_steps,
                                            unlearn_lr=args.unlearn_lr,
                                            batch_size=args.batch_size,
                                            )

    trainer.train()
    if args.save_model:
        trainer.save_model(save_path)
        processor.save_pretrained(save_path)

    ###################################### Predict Results ####################################
    test_pred = generate_predictions_for_metrics(trainer.model, test_dataset, processor)

    if args.attack == "codebookv2":
        method_name = "codebook"
    else:
        method_name = "badnet"

    save_results(
        test_pred,
        processor,
        test_dataset,
        method_name=method_name,
        save_path=f"{res_path}/res_test.json",
        label=f"test"
    )

    test_clean_pred = generate_predictions_for_metrics(trainer.model, test_clean_dataset, processor)
    save_results(
        test_clean_pred,
        processor,
        test_clean_dataset,
        save_path=f"{res_path}/res_test_clean.json",
        label=f"test_clean"
    )

    train_subset = train_dataset.select(range(min(1600, len(train_dataset))))
    train_pred = generate_predictions_for_metrics(trainer.model, train_subset, processor)
    save_results(
        train_pred,
        processor,
        train_subset,
        save_path=f"{res_path}/res_train.json",
        label=f"train"
    )

if __name__ == "__main__":
    main()

