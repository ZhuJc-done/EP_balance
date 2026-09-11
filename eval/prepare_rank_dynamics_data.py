#!/usr/bin/env python3
"""Create a deterministic, fixed-token-budget JSONL subset for routing capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help='Source JSONL with a "text" field')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _decode(tokenizer, token_ids: list[int]) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        return backend.decode(token_ids, skip_special_tokens=False)
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _can_reuse(
    output: Path,
    manifest_path: Path,
    *,
    source: Path,
    tokenizer_model: str,
    token_budget: int,
) -> bool:
    if not output.is_file() or output.stat().st_size == 0 or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        manifest.get("source") == str(source)
        and manifest.get("source_size_bytes") == source.stat().st_size
        and manifest.get("tokenizer") == tokenizer_model
        and manifest.get("requested_tokens") == token_budget
        and manifest.get("accepted_tokens") == token_budget
    )


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = output.with_suffix(".manifest.json")

    if args.token_budget <= 0:
        raise ValueError("--token-budget must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"source JSONL not found: {source}")
    if source == output:
        raise ValueError("--input and --output must be different files")
    if not args.force and _can_reuse(
        output,
        manifest_path,
        source=source,
        tokenizer_model=args.tokenizer_model,
        token_budget=args.token_budget,
    ):
        print(f"[prepare_rank_dynamics_data] reuse {output}")
        return

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the tokenizer dependency: pip install transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    accepted_tokens = 0
    accepted_documents = 0
    scanned_documents = 0

    try:
        with source.open(encoding="utf-8") as input_stream, temporary.open(
            "w", encoding="utf-8"
        ) as output_stream:
            for line_number, line in enumerate(input_stream, start=1):
                if accepted_tokens >= args.token_budget:
                    break
                scanned_documents += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
                text = record.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue

                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if not token_ids:
                    continue
                remaining = args.token_budget - accepted_tokens
                if len(token_ids) > remaining:
                    token_ids = token_ids[:remaining]
                    text = _decode(tokenizer, token_ids)
                output_stream.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                accepted_tokens += len(token_ids)
                accepted_documents += 1
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    if accepted_tokens != args.token_budget:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{source} supplied {accepted_tokens} tokens, fewer than "
            f"the requested {args.token_budget}"
        )
    temporary.replace(output)

    manifest = {
        "source": str(source),
        "source_size_bytes": source.stat().st_size,
        "output": str(output),
        "tokenizer": args.tokenizer_model,
        "requested_tokens": args.token_budget,
        "accepted_tokens": accepted_tokens,
        "accepted_documents": accepted_documents,
        "scanned_documents": scanned_documents,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "[prepare_rank_dynamics_data] "
        f"{accepted_documents} documents / {accepted_tokens} tokens -> {output}"
    )


if __name__ == "__main__":
    main()
