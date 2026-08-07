#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

from relearning_corpus import LANGUAGES, read_rows, split_rows


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=args.revision, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
        device_map={"": "cuda:0"},
    )
    model = PeftModel.from_pretrained(model, str(args.adapter)).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), use_fast=True)

    results = {}
    with torch.inference_mode():
        for language in LANGUAGES:
            _, heldout = split_rows(read_rows(args.corpus_zip, language),
                                    args.seed, args.validation_fraction)
            weighted, tokens, truncated = 0.0, 0, 0
            for row in heldout:
                text = row["text"] + (tokenizer.eos_token or "")
                full = tokenizer(text, add_special_tokens=True)
                truncated += int(len(full["input_ids"]) > args.max_length)
                batch = tokenizer(text, truncation=True, max_length=args.max_length,
                                  return_tensors="pt").to("cuda:0")
                n = max(0, batch["input_ids"].shape[1] - 1)
                if not n:
                    continue
                loss = model(**batch, labels=batch["input_ids"]).loss.float().item()
                if not math.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss for {language}")
                weighted += loss * n
                tokens += n

            nll = weighted / tokens
            results[language] = {
                "heldout_passages": len(heldout),
                "predicted_tokens": tokens,
                "truncated_passages": truncated,
                "nll": nll,
                "perplexity": math.exp(nll),
            }

    return {
        "base_model": args.base_model,
        "revision": args.revision,
        "adapter": str(args.adapter),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "max_length": args.max_length,
        "languages": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--corpus-zip", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = evaluate(args)
    save_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
