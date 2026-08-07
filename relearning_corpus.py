import hashlib
import json
import math
import zipfile
from pathlib import Path

LANGUAGES = ("en", "es", "ar", "ja")


def read_rows(archive: Path, language: str) -> list[dict]:
    member = f"relearn.{language}.jsonl"
    with zipfile.ZipFile(archive) as zf:
        if member not in zf.namelist():
            raise ValueError(f"archive missing {member}")
        rows = [json.loads(line) for line in zf.read(member).decode("utf-8").splitlines()
                if line.strip()]

    passage_ids = []
    for number, row in enumerate(rows, 1):
        if not isinstance(row.get("passage_id"), str) or not row["passage_id"]:
            raise ValueError(f"{member}:{number}: invalid passage_id")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"{member}:{number}: invalid text")
        row["text"] = row["text"].strip()
        passage_ids.append(row["passage_id"])
    if len(passage_ids) != len(set(passage_ids)):
        raise ValueError(f"duplicate passage_id in {member}")
    return rows


def validate_alignment(archive: Path, languages=LANGUAGES) -> None:
    rows = {language: read_rows(archive, language) for language in languages}
    reference = {row["passage_id"] for row in rows[languages[0]]}
    for language in languages[1:]:
        if {row["passage_id"] for row in rows[language]} != reference:
            raise ValueError(f"passage IDs are not aligned for {language}")


def split_rows(rows: list[dict], seed: int, fraction: float) -> tuple[list[dict], list[dict]]:
    if not 0 < fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    ids = sorted({row["passage_id"] for row in rows},
                 key=lambda pid: hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest())
    heldout_ids = set(ids[:max(1, math.ceil(len(ids) * fraction))])
    return ([row for row in rows if row["passage_id"] not in heldout_ids],
            [row for row in rows if row["passage_id"] in heldout_ids])
