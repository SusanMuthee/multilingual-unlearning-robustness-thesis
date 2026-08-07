from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from relearning_corpus import LANGUAGES, read_rows, split_rows


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = {"model", "revision", "corpus_zip", "condition", "seed", "max_length",
                "epochs", "learning_rate", "micro_batch_size", "gradient_accumulation_steps",
                "output_dir", "lora_rank", "lora_alpha", "lora_dropout", "target_modules"}
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"missing configuration keys: {missing}")
    if cfg["condition"] not in LANGUAGES:
        raise ValueError(f"condition must be one of {LANGUAGES}")
    if not 64 <= int(cfg["max_length"]) <= 2048:
        raise ValueError("max_length must be between 64 and 2048")
    if int(cfg["lora_rank"]) <= 0 or int(cfg["lora_alpha"]) <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0 <= float(cfg["lora_dropout"]) < 1:
        raise ValueError("LoRA dropout must be in [0,1)")
    return cfg


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg: dict, config_path: Path) -> None:
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("a BF16-capable CUDA GPU is required")
    archive = Path(cfg["corpus_zip"]).resolve()
    output = Path(cfg["output_dir"]).resolve()
    rows = read_rows(archive, cfg["condition"])
    train_rows, held_rows = split_rows(rows, int(cfg["seed"]),
                                       float(cfg.get("validation_fraction", 0.2)))
    set_seed(int(cfg["seed"]))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"], revision=cfg["revision"],
        max_seq_length=int(cfg["max_length"]), dtype=torch.bfloat16,
        load_in_4bit=False, load_in_8bit=False, full_finetuning=False,
        fast_inference=False, trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model, r=int(cfg["lora_rank"]), lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]), bias="none",
        target_modules=list(cfg["target_modules"]),
        use_gradient_checkpointing="unsloth", random_state=int(cfg["seed"]),
        use_rslora=False, loftq_config=None,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable <= 0 or trainable >= total:
        raise RuntimeError(f"LoRA invariant failed: trainable={trainable}, total={total}")

    eos = tokenizer.eos_token or ""
    max_length = int(cfg["max_length"])
    def encode(batch):
        encoded = tokenizer([x + eos for x in batch["text"]], truncation=True,
                            max_length=max_length, padding=False, add_special_tokens=True)
        encoded["labels"] = [x.copy() for x in encoded["input_ids"]]
        return encoded

    train_ds = Dataset.from_list(train_rows).shuffle(seed=int(cfg["seed"]))
    held_ds = Dataset.from_list(held_rows)
    train_tok = train_ds.map(encode, batched=True, remove_columns=train_ds.column_names)
    held_tok = held_ds.map(encode, batched=True, remove_columns=held_ds.column_names)

    class Collator:
        def __call__(self, features):
            labels = [x.pop("labels") for x in features]
            batch = tokenizer.pad(features, padding=True, return_tensors="pt")
            width = batch["input_ids"].shape[1]
            batch["labels"] = torch.tensor(
                [x + [-100] * (width - len(x)) for x in labels], dtype=torch.long)
            return batch

    args = TrainingArguments(
        output_dir=str(output / "trainer_output"),
        num_train_epochs=float(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        per_device_train_batch_size=int(cfg["micro_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        per_device_eval_batch_size=1, bf16=True, tf32=True,
        optim="adamw_torch_fused", lr_scheduler_type="cosine",
        logging_strategy="no",
        eval_strategy="epoch",
        save_strategy="no", report_to=[], max_grad_norm=1.0,
        seed=int(cfg["seed"]), data_seed=int(cfg["seed"]),
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_tok,
                      eval_dataset=held_tok,
                      data_collator=Collator())
    training_result = trainer.train()
    evaluation_metrics = trainer.evaluate()

    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)

    save_json(output / "training_metadata.json", {
        "experiment_id": cfg["experiment_id"],
        "model_family": cfg["family"],
        "unlearning_method": cfg["method"],
        "relearning_language": cfg["condition"],
        "model": cfg["model"],
        "revision": cfg["revision"],
        "seed": int(cfg["seed"]),
        "training_examples": len(train_rows),
        "heldout_examples": len(held_rows),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
        "configuration_sha256": sha256(config_path),
        "corpus_sha256": sha256(archive),
        "training_metrics": dict(training_result.metrics),
        "heldout_metrics": evaluation_metrics,
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    ns = p.parse_args()
    cfg = load_config(ns.config.resolve())
    train(cfg, ns.config.resolve())


if __name__ == "__main__":
    main()
