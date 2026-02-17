#!/usr/bin/env python3
"""Compact docs/js/data.js by stripping whitespace from the JSON payload."""

import json
import sys
from pathlib import Path

DATA_JS = Path(__file__).resolve().parent.parent / "docs" / "js" / "data.js"

def main():
    content = DATA_JS.read_text()
    start = content.index("{")
    raw_json = content[start:].rstrip().rstrip(";")
    data = json.loads(raw_json)
    DATA_JS.write_text(
        "const SWEEP_DATA=" + json.dumps(data, separators=(",", ":")) + ";"
    )
    size_mib = DATA_JS.stat().st_size / 1024 / 1024
    print(f"  Compacted data.js: {size_mib:.1f} MiB")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"compact_data.py: {exc}", file=sys.stderr)
        sys.exit(1)
