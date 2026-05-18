from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_config
from .graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run entity-only equipment NER evaluation.")
    parser.add_argument("--input", dest="input_md", default=None, help="Input OCR Markdown file.")
    parser.add_argument("--output", dest="output_dir", default=None, help="Output directory.")
    parser.add_argument("--env", dest="env_path", default=None, help="Optional .env path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.input_md, args.output_dir, args.env_path)
        input_path = Path(config.input_md)
        if not input_path.exists():
            print(f"distill_ner failed: input Markdown not found: {config.input_md}", file=sys.stderr)
            return 1
        graph = build_graph(config)
        final_state = graph.invoke(
            {
                "input_md": config.input_md,
                "output_dir": config.output_dir,
                "errors": [],
            }
        )
        compare = final_state.get("summary_compare") or {}
        if compare:
            print(json.dumps(compare, ensure_ascii=False, indent=2))
        if final_state.get("errors"):
            print("\nErrors:", file=sys.stderr)
            for error in final_state["errors"]:
                print(f"- {error}", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:
        print(f"distill_ner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
