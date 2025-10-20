import json
import hashlib
from pathlib import Path
from typing import Tuple

import ijson
from pydantic import ValidationError

try:
    import commentjson  # only needed if your file truly has // comments
except Exception:  # pragma: no cover
    commentjson = None

from models import Professor

RAW_DIR = Path("/home/sina-farsangi/Desktop/sina/RAG_Bcs_prj/data/raw")
OUT_VALID = Path("/home/sina-farsangi/Desktop/sina/RAG_Bcs_prj/data/validated")
OUT_REJECTS = Path("/home/sina-farsangi/Desktop/sina/RAG_Bcs_prj/data/rejects")
REPORTS = Path("/home/sina-farsangi/Desktop/sina/RAG_Bcs_prj/reports")
for d in (OUT_VALID, OUT_REJECTS, REPORTS):
    d.mkdir(parents=True, exist_ok=True)


def sha1_of_obj(obj) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def write_ok(fh, obj):
    rec = {"_id": sha1_of_obj(obj), **obj}
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_bad(fh, err, obj):
    fh.write(json.dumps({"error": err, "raw": obj}, ensure_ascii=False) + "\n")


def validate_top_level_array(input_path: Path, basename: str) -> Tuple[int, int]:
    """Stream a top-level JSON array with ijson. Fallback to commentjson if comments exist."""
    total = good = bad = 0
    valid_path = OUT_VALID / f"{basename}.jsonl"
    bad_path = OUT_REJECTS / f"{basename}_bad.jsonl"

    with open(valid_path, "w", encoding="utf-8") as f_ok, \
         open(bad_path, "w", encoding="utf-8") as f_bad, \
         open(input_path, "r", encoding="utf-8") as f_in:

        try:
            for obj in ijson.items(f_in, "item"):
                total += 1
                if not isinstance(obj, dict):
                    bad += 1
                    write_bad(f_bad, "non-object item at top level", obj)
                    continue
                try:
                    # Populate by alias → our pythonic fields don't need to match JSON keys
                    Professor.model_validate(obj)
                    write_ok(f_ok, obj)
                    good += 1
                except ValidationError as ve:
                    bad += 1
                    write_bad(f_bad, ve.errors(), obj)

        except ijson.common.IncompleteJSONError as e:
            # Probably JSONC with comments. Try commentjson if available.
            if commentjson is None:
                raise RuntimeError(
                    "File looks like JSONC (with comments) but 'commentjson' is not installed. "
                    "Install with: pip install commentjson"
                ) from e

            f_in.seek(0)
            data = commentjson.load(f_in)  # expects a top-level list
            if not isinstance(data, list):
                raise RuntimeError("JSONC fallback: top-level structure is not a list/array.")

            for obj in data:
                total += 1
                if not isinstance(obj, dict):
                    bad += 1
                    write_bad(f_bad, "non-object item at top level", obj)
                    continue
                try:
                    Professor.model_validate(obj)
                    write_ok(f_ok, obj)
                    good += 1
                except ValidationError as ve:
                    bad += 1
                    write_bad(f_bad, ve.errors(), obj)

    # summary
    report = {"file": str(input_path), "total": total, "valid": good, "rejected": bad}
    with open(REPORTS / f"{basename}_validation_summary.json", "w", encoding="utf-8") as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2)
    return good, bad


if __name__ == "__main__":
    # Adjust these names as present in data/raw/
    candidates = [
        ("professors_norm.json", "professors_norm"),
        ("professors_mini.jsonc", "professors_mini"),
    ]
    for filename, base in candidates:
        path = RAW_DIR / filename
        if not path.exists():
            print(f"Skip (not found): {filename}")
            continue
        g, b = validate_top_level_array(path, base)
        print(f"Validated {filename}: valid={g}, rejected={b} → see reports/{base}_validation_summary.json")