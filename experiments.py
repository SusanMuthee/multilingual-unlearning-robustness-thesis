#!/usr/bin/env python3
"""Run the fixed 2-model x 3-method x 4-language experiment matrix."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from relearning_corpus import validate_alignment

HERE = Path(__file__).resolve().parent


def execute(arguments: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(arguments, cwd=HERE, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def benchmark(model: str, revision: str, output: Path, settings: dict,
              seed: int, adapter: Path | None = None) -> None:
    model_args = f"pretrained={model},revision={revision},dtype=bfloat16"
    if adapter is not None:
        model_args += f",peft={adapter}"
    execute([
        "lm_eval", "--model", "hf", "--model_args", model_args,
        "--tasks", settings["tasks"],
        "--num_fewshot", str(settings["num_fewshot"]),
        "--batch_size", str(settings["batch_size"]),
        "--seed", str(seed), "--output_path", str(output),
    ], output.parent / f"{output.name}.log")


def make_training_config(study: dict, experiment: dict, language: str,
                         output: Path, path: Path) -> None:
    config = dict(study["relearning"])
    config.update({
        "experiment_id": experiment["id"],
        "family": experiment["family"],
        "method": experiment["method"],
        "model": experiment["unlearned_model"],
        "revision": experiment["unlearned_revision"],
        "corpus_zip": study["corpus_zip"],
        "condition": language,
        "seed": study["seed"],
        "output_dir": str(output),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_study(config_dir: Path) -> dict:
    study = read_json(config_dir / "study.json")
    study["relearning"] = read_json(config_dir / "relearning.json")
    study["evaluation"] = read_json(config_dir / "evaluation.json")
    study.update(read_json(config_dir / "models.json"))
    return study


def run_study(config_dir: Path) -> None:
    study = load_study(config_dir)
    validate_alignment(Path(study["corpus_zip"]), tuple(study["languages"]))
    root = Path(study["results_dir"])

    for experiment in study["experiments"]:
        experiment_root = root / experiment["id"]

        if "reuse_original_from" not in experiment:
            benchmark(experiment["base_model"], experiment["base_revision"],
                      experiment_root / "baseline_original", study["evaluation"], study["seed"])

        benchmark(experiment["unlearned_model"], experiment["unlearned_revision"],
                  experiment_root / "baseline_unlearned", study["evaluation"], study["seed"])

        for language in study["languages"]:
            output = experiment_root / language
            training_config = root / "configs" / f"{experiment['id']}__{language}.json"
            make_training_config(study, experiment, language, output, training_config)
            execute([sys.executable, "relearn.py", "train", "--config", str(training_config)],
                    root / "logs" / f"{experiment['id']}__{language}__train.log")

            adapter = output / "adapter"
            execute([
                sys.executable, "evaluate.py",
                "--base-model", experiment["unlearned_model"],
                "--revision", experiment["unlearned_revision"],
                "--adapter", str(adapter), "--corpus-zip", study["corpus_zip"],
                "--seed", str(study["seed"]),
                "--max-length", str(study["evaluation"]["max_length"]),
                "--output", str(output / "parallel_after.json"),
            ], root / "logs" / f"{experiment['id']}__{language}__parallel.log")

            benchmark(experiment["unlearned_model"], experiment["unlearned_revision"],
                      output / "benchmark", study["evaluation"], study["seed"], adapter)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path,
                        default=Path(__file__).with_name("configs"))
    args = parser.parse_args()
    run_study(args.config_dir.resolve())


if __name__ == "__main__":
    main()
