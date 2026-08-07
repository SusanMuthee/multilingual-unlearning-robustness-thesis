#!/usr/bin/env python3

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


def make_training_config(settings: dict, experiment: dict, language: str,
                         output: Path, path: Path) -> None:
    config = dict(settings["relearning"])
    config.update({
        "experiment_id": experiment["id"],
        "family": experiment["family"],
        "method": experiment["method"],
        "model": experiment["unlearned_model"],
        "revision": experiment["unlearned_revision"],
        "corpus_zip": settings["corpus_zip"],
        "condition": language,
        "seed": settings["seed"],
        "output_dir": str(output),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiments(config_dir: Path) -> dict:
    settings = read_json(config_dir / "experiments.json")
    settings["relearning"] = read_json(config_dir / "relearning.json")
    settings["evaluation"] = read_json(config_dir / "evaluation.json")
    settings.update(read_json(config_dir / "models.json"))
    return settings


def run_experiments(config_dir: Path) -> None:
    settings = load_experiments(config_dir)
    validate_alignment(Path(settings["corpus_zip"]), tuple(settings["languages"]))
    root = Path(settings["results_dir"])

    for family_name, family in settings["families"].items():
        family_root = root / family_name
        benchmark(family["base_model"], family["base_revision"],
                  family_root / "baseline_original", settings["evaluation"], settings["seed"])

        for method, checkpoint in family["unlearned"].items():
            experiment = {
                "id": f"{family_name}_{method}",
                "family": family_name,
                "method": method,
                "unlearned_model": checkpoint["model"],
                "unlearned_revision": checkpoint["revision"],
            }
            experiment_root = root / experiment["id"]
            benchmark(checkpoint["model"], checkpoint["revision"],
                      experiment_root / "baseline_unlearned",
                      settings["evaluation"], settings["seed"])

            for language in settings["languages"]:
                output = experiment_root / language
                training_config = root / "configs" / f"{experiment['id']}__{language}.json"
                make_training_config(settings, experiment, language, output, training_config)
                execute([sys.executable, "relearn.py", "--config", str(training_config)],
                        root / "logs" / f"{experiment['id']}__{language}__train.log")

                adapter = output / "adapter"
                execute([
                    sys.executable, "evaluate.py",
                    "--base-model", checkpoint["model"],
                    "--revision", checkpoint["revision"],
                    "--adapter", str(adapter), "--corpus-zip", settings["corpus_zip"],
                    "--seed", str(settings["seed"]),
                    "--max-length", str(settings["evaluation"]["max_length"]),
                    "--output", str(output / "parallel_after.json"),
                ], root / "logs" / f"{experiment['id']}__{language}__parallel.log")

                benchmark(checkpoint["model"], checkpoint["revision"],
                          output / "benchmark", settings["evaluation"], settings["seed"], adapter)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path,
                        default=Path(__file__).with_name("configs"))
    args = parser.parse_args()
    run_experiments(args.config_dir.resolve())


if __name__ == "__main__":
    main()
