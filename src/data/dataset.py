
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "record_id",
    "text",
    "source_type",
    "source_title",
    "source_url",
    "license",
    "topic",
    "collection_date",
    "quality_status",
}

VALID_SOURCE_TYPES = {"synthetic", "real_world"}


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a JSONL corpus file."""
    corpus_path = Path(path)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with corpus_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            missing_fields = REQUIRED_FIELDS - record.keys()
            if missing_fields:
                raise ValueError(
                    f"Line {line_number} is missing fields: "
                    f"{sorted(missing_fields)}"
                )

            record_id = record["record_id"]

            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(
                    f"Line {line_number} has an invalid record_id."
                )

            if record_id in seen_ids:
                raise ValueError(
                    f"Duplicate record_id '{record_id}' "
                    f"on line {line_number}."
                )

            if not isinstance(record["text"], str) or not record["text"].strip():
                raise ValueError(
                    f"Line {line_number} has empty or invalid text."
                )

            if record["source_type"] not in VALID_SOURCE_TYPES:
                raise ValueError(
                    f"Line {line_number} has an invalid source_type."
                )

            seen_ids.add(record_id)
            records.append(record)

    return records
