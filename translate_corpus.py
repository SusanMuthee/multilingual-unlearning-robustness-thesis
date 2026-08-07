import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path


PRESERVATION_PATTERNS = {
    "numbers": re.compile(r"(?<!\w)[+-]?(?:\d+[.,]?)+%?(?!\w)"),
    "urls": re.compile(r"https?://\S+"),
    "code_like": re.compile(
        r"\b(?:[A-Z][A-Z0-9_-]{2,}|[A-Za-z0-9_.-]+\.(?:exe|dll|py|sh|ps1))\b"
    ),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def preservation_flags(source: str, translation: str) -> dict:
    flags = {}
    for name, pattern in PRESERVATION_PATTERNS.items():
        source_items = sorted(set(pattern.findall(source)))
        translated_items = sorted(set(pattern.findall(translation)))
        if source_items != translated_items:
            flags[name] = {"source": source_items, "translation": translated_items}
    return flags


def validate_master(rows: list[dict]) -> None:
    identifiers = [row.get("passage_id") for row in rows]
    if not rows or any(not isinstance(identifier, str) or not identifier
                       for identifier in identifiers):
        raise ValueError("every master row requires a passage_id")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("master passage_id values must be unique")
    if any(not isinstance(row.get("text"), str) or not row["text"].strip()
           for row in rows):
        raise ValueError("every master row requires non-empty text")


def translate(config: dict) -> None:
    import torch
    from sacrebleu.metrics import CHRF
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    master_path = Path(config["master_jsonl"])
    output_dir = Path(config["output_dir"])
    archive_path = Path(config["archive"])
    rows = read_jsonl(master_path)
    validate_master(rows)

    model_id = config["model"]
    revision = config["revision"]
    languages = config["languages"]
    maximum_length = int(config["max_length"])
    number_of_beams = int(config["num_beams"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, src_lang=languages["en"]
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float16
    ).to("cuda").eval()
    chrf2 = CHRF(word_order=2)

    @torch.inference_mode()
    def translate_text(text: str, source_language: str, target_language: str) -> str:
        tokenizer.src_lang = source_language
        encoded = tokenizer(text, return_tensors="pt", truncation=False).to("cuda")
        if encoded["input_ids"].shape[1] > maximum_length:
            raise ValueError("source passage exceeds the configured NLLB length")
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_language),
            max_length=maximum_length,
            num_beams=number_of_beams,
            do_sample=False,
        )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    qa_summary = {}
    for language, language_code in languages.items():
        corpus_rows = []
        qa_rows = []
        for row in rows:
            source_text = row["text"].strip()
            if language == "en":
                translated_text = source_text
                roundtrip_text = source_text
            else:
                translated_text = translate_text(
                    source_text, languages["en"], language_code
                )
                roundtrip_text = translate_text(
                    translated_text, language_code, languages["en"]
                )

            corpus_rows.append({
                **row,
                "language": language,
                "source_language": "en",
                "translation_system": "identity" if language == "en" else model_id,
                "translation_revision": None if language == "en" else revision,
                "text": translated_text,
            })
            qa_rows.append({
                "passage_id": row["passage_id"],
                "language": language,
                "source_sha256": sha256_text(source_text),
                "translation_sha256": sha256_text(translated_text),
                "roundtrip_chrf2": float(
                    chrf2.sentence_score(roundtrip_text, [source_text]).score
                ),
                "preservation_flags": preservation_flags(source_text, translated_text),
                "automated_qa_only": True,
            })

        corpus_path = output_dir / f"relearn.{language}.jsonl"
        qa_path = output_dir / f"qa.{language}.jsonl"
        write_jsonl(corpus_path, corpus_rows)
        write_jsonl(qa_path, qa_rows)
        files[language] = {
            "path": corpus_path.name,
            "sha256": sha256_file(corpus_path),
            "record_count": len(corpus_rows),
            "qa_path": qa_path.name,
            "qa_sha256": sha256_file(qa_path),
        }
        qa_summary[language] = {
            "minimum_roundtrip_chrf2": min(row["roundtrip_chrf2"] for row in qa_rows),
            "mean_roundtrip_chrf2": sum(row["roundtrip_chrf2"] for row in qa_rows)
            / len(qa_rows),
            "preservation_flag_count": sum(
                bool(row["preservation_flags"]) for row in qa_rows
            ),
        }

    manifest = {
        "corpus_name": archive_path.stem,
        "master_sha256": sha256_file(master_path),
        "translation_model": model_id,
        "translation_revision": revision,
        "languages": list(languages),
        "max_length": maximum_length,
        "num_beams": number_of_beams,
        "files": files,
        "qa_summary": qa_summary,
        "answer_leakage_screened": False,
        "quality_statement": "Machine translated and automatically screened; not human verified.",
    }
    manifest_path = output_dir / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(master_path, master_path.name)
        archive.write(manifest_path, manifest_path.name)
        for metadata in files.values():
            archive.write(output_dir / metadata["path"], metadata["path"])
            archive.write(output_dir / metadata["qa_path"], metadata["qa_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/translation.json"))
    arguments = parser.parse_args()
    translate(read_json(arguments.config))


if __name__ == "__main__":
    main()
