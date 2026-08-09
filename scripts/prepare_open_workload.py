#!/usr/bin/env python3
"""Build domain-specific open workloads and optionally preprocess them for Megatron-LM.

The output JSONL contains one ``{"text": ...}`` document per source record.
Task-style adapters deliberately exclude answers/reference solutions; pretraining
corpora retain their source text. The generated files are suitable for real
Megatron training as well as routing-analysis workloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Iterator


WORKLOADS = {
    "dapo_math": {
        "dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
        "config": None,
        "split": "train",
    },
    "fineweb": {
        "dataset": "HuggingFaceFW/fineweb",
        "config": "sample-10BT",
        "split": "train",
        "text_fields": ("text",),
    },
    # Current datasets releases no longer execute allenai/dolma's loading
    # script. Resolve the official v1.6 sample URL list and stream JSON instead.
    "dolma": {
        "dataset": "allenai/dolma",
        "config": "v1_6-sample",
        "split": "train",
        "text_fields": ("text",),
        "loader": "dolma_url_list",
        "url_list": "urls/v1_6-sample.txt",
    },
    "fineweb2_zh": {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config": "cmn_Hani",
        "split": "train",
        "text_fields": ("text",),
    },
    # Round-robin the four source directories and divide --token-budget
    # equally between them so one language cannot dominate the code workload.
    "starcoder": {
        "dataset": "bigcode/starcoderdata",
        "config": None,
        "split": "train",
        "text_fields": ("content",),
        "data_dirs": ("python", "cpp", "java", "rust"),
        "balance_sources": True,
    },
    # peS2o also uses a retired loading script. One v2 shard is already much
    # larger than this experiment's token budget, so stream that file directly.
    "pes2o": {
        "dataset": "allenai/peS2o",
        "config": "v2",
        "split": "train",
        "text_fields": ("text",),
        "loader": "json",
        "data_files": (
            "hf://datasets/allenai/peS2o/data/v2/train-00000-of-00020.json.gz",
        ),
    },
    # Legacy UltraEP-style comparison workloads remain available, but are not
    # part of the independent primary panel above.
    "openscience": {
        "dataset": "nvidia/OpenScience",
        "config": "OS-Q3-235B-4",
        "split": "train",
    },
    "codeforces": {
        "dataset": "open-r1/codeforces",
        "config": "verifiable-prompts",
        "split": "train",
    },
    "swe_bench": {
        "dataset": "SWE-bench/SWE-bench",
        "config": "default",
        "split": "test",
    },
    "gpqa": {
        "dataset": "Idavidrein/gpqa",
        "config": "gpqa_main",
        "split": "train",
    },
    "wikitext": {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
    },
    # LongBench v1 is read directly from data.zip because recent versions of
    # ``datasets`` no longer execute the repository's dataset loading script.
    "longbench": {
        "dataset": "THUDM/LongBench",
        "config": None,
        "split": "test",
    },
}

CORE_WORKLOADS = (
    "dapo_math",
    "fineweb",
    "dolma",
    "fineweb2_zh",
    "starcoder",
    "pes2o",
)

DEFAULT_LONGBENCH_TASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "gov_report",
    "qmsum",
    "multi_news",
    "lcc",
    "repobench-p",
)


def _format_messages(messages: object, tokenizer, use_chat_template: bool) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    normalized = [
        {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
        for message in messages
        if isinstance(message, dict) and message.get("content")
    ]
    if not normalized:
        return ""
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n\n".join(
        f"{message['role'].capitalize()}:\n{message['content']}" for message in normalized
    )


def _first_present(example: dict, *names: str) -> str:
    for name in names:
        value = example.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _decode_token_ids(tokenizer, token_ids: list[int]) -> str:
    """Decode through the fast-tokenizer backend when available."""
    ids = [int(token_id) for token_id in token_ids]
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        return backend.decode(ids, skip_special_tokens=False)
    return tokenizer.decode(ids, skip_special_tokens=False)


def extract_prompt(
    workload: str,
    example: dict,
    tokenizer,
    *,
    use_chat_template: bool,
    rng: random.Random,
    text_fields: tuple[str, ...] | None = None,
) -> str:
    """Extract only the model input from one source record."""
    if workload == "dapo_math":
        return _format_messages(example.get("prompt"), tokenizer, use_chat_template)
    fields = text_fields
    if fields is None:
        fields = tuple(WORKLOADS.get(workload, {}).get("text_fields", ()))
    if fields:
        return _first_present(example, *fields)
    if workload == "openscience":
        return _first_present(example, "input")
    if workload == "codeforces":
        return _first_present(example, "prompt")
    if workload == "swe_bench":
        return _first_present(example, "problem_statement")
    if workload == "wikitext":
        return _first_present(example, "text")
    if workload == "longbench":
        context = _first_present(example, "context")
        instruction = _first_present(example, "input", "question")
        if not context or not instruction:
            return ""
        return f"Context:\n{context}\n\nQuestion or instruction:\n{instruction}\n\nAnswer:"
    if workload == "gpqa":
        question = _first_present(example, "Question", "question", "problem")
        options = [
            _first_present(example, "Correct Answer", "correct_answer"),
            _first_present(example, "Incorrect Answer 1", "incorrect_answer_1"),
            _first_present(example, "Incorrect Answer 2", "incorrect_answer_2"),
            _first_present(example, "Incorrect Answer 3", "incorrect_answer_3"),
        ]
        options = [option for option in options if option]
        if not question or len(options) != 4:
            return ""
        # The official columns place the correct answer first. Shuffle to avoid
        # leaking that artifact into the routing workload.
        rng.shuffle(options)
        rendered = "\n".join(f"{chr(65 + idx)}. {option}" for idx, option in enumerate(options))
        return f"{question}\n\n{rendered}\n\nSelect the best answer."
    raise ValueError(f"unsupported workload: {workload}")


def _iter_longbench(tasks: tuple[str, ...]) -> Iterator[dict]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("LongBench preparation requires huggingface_hub") from exc

    archive = hf_hub_download(
        repo_id="THUDM/LongBench",
        filename="data.zip",
        repo_type="dataset",
    )
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        for task in tasks:
            suffix = f"data/{task}.jsonl"
            matches = [name for name in members if name.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected exactly one LongBench member ending in {suffix!r}, got {matches}"
                )
            with zf.open(matches[0]) as source:
                for raw_line in source:
                    example = json.loads(raw_line.decode("utf-8"))
                    example["_longbench_task"] = task
                    yield example


def _iter_hf_dataset(
    dataset_id: str,
    config: str | None,
    split: str,
    *,
    seed: int,
    shuffle_buffer: int,
    data_dir: str | None = None,
    data_files: str | list[str] | tuple[str, ...] | None = None,
) -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the dataset dependency first: pip install datasets") from exc

    args = [dataset_id]
    if config:
        args.append(config)
    kwargs = {"split": split, "streaming": True}
    if data_dir:
        kwargs["data_dir"] = data_dir
    if data_files:
        kwargs["data_files"] = data_files
    dataset = load_dataset(*args, **kwargs)
    if shuffle_buffer > 1:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return dataset


def _dolma_urls(dataset_id: str, filename: str, limit: int) -> list[str]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Dolma preparation requires huggingface_hub") from exc

    url_list = Path(
        hf_hub_download(dataset_id, filename, repo_type="dataset")
    ).read_text(encoding="utf-8")
    urls = [line.strip() for line in url_list.splitlines() if line.strip()]
    if not urls:
        raise RuntimeError(f"Dolma URL list is empty: {dataset_id}/{filename}")
    return urls if limit == 0 else urls[:limit]


def _resolve_sources(args, spec: dict, dataset_id: str, config: str | None, split: str) -> list[dict]:
    """Resolve one or more physical loaders for a logical workload."""
    if args.dataset_id or args.dataset_data_dir or args.data_file:
        return [
            {
                "name": args.dataset_data_dir or "default",
                "loader": dataset_id,
                "config": config,
                "split": split,
                "data_dir": args.dataset_data_dir,
                "data_files": tuple(args.data_file),
            }
        ]

    if spec.get("loader") == "dolma_url_list":
        urls = _dolma_urls(dataset_id, spec["url_list"], args.source_file_limit)
        return [
            {
                "name": "official-v1_6-sample",
                "loader": "json",
                "config": None,
                "split": split,
                "data_dir": None,
                "data_files": tuple(urls),
            }
        ]

    data_dirs = tuple(spec.get("data_dirs", ()))
    if data_dirs:
        return [
            {
                "name": data_dir,
                "loader": dataset_id,
                "config": config,
                "split": split,
                "data_dir": data_dir,
                "data_files": (),
            }
            for data_dir in data_dirs
        ]

    return [
        {
            "name": "default",
            "loader": spec.get("loader", dataset_id),
            "config": None if spec.get("loader") else config,
            "split": split,
            "data_dir": None,
            "data_files": tuple(spec.get("data_files", ())),
        }
    ]


def _balanced_token_budgets(total: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("source count must be positive")
    base, remainder = divmod(total, count)
    return [base + (index < remainder) for index in range(count)]


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return int(ordered[index])


def _write_workload(args, tokenizer) -> tuple[Path, dict]:
    spec = WORKLOADS[args.workload]
    dataset_id = args.dataset_id or spec["dataset"]
    config = args.dataset_config if args.dataset_config is not None else spec["config"]
    split = args.split or spec["split"]
    text_fields = (
        (args.text_field,)
        if args.text_field
        else tuple(spec.get("text_fields", ()))
    )
    rng = random.Random(args.seed)

    if args.workload == "longbench":
        tasks = tuple(task.strip() for task in args.longbench_tasks.split(",") if task.strip())
        examples = list(_iter_longbench(tasks))
        rng.shuffle(examples)
        sources = [
            {
                "name": "longbench",
                "loader": "THUDM/LongBench:data.zip",
                "config": ",".join(tasks),
                "split": split,
                "data_dir": None,
                "data_files": (),
                "iterator": iter(examples),
            }
        ]
    else:
        sources = _resolve_sources(args, spec, dataset_id, config, split)
        for source in sources:
            source["iterator"] = iter(
                _iter_hf_dataset(
                    source["loader"],
                    source["config"],
                    source["split"],
                    seed=args.seed,
                    shuffle_buffer=args.shuffle_buffer,
                    data_dir=source["data_dir"],
                    data_files=source["data_files"],
                )
            )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_jsonl).expanduser().resolve() if args.output_jsonl else (
        output_dir / f"{args.workload}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    token_lengths: list[int] = []
    skipped_empty = 0
    skipped_short = 0
    skipped_long = 0
    skipped_duplicate = 0
    skipped_budget = 0
    accepted_tokens = 0
    source_rows_scanned = 0
    seen_prompts: set[bytes] = set()

    balance_sources = bool(
        spec.get("balance_sources")
        and not args.no_balance_sources
        and len(sources) > 1
        and args.token_budget
    )
    token_budgets = (
        _balanced_token_budgets(args.token_budget, len(sources))
        if balance_sources
        else [None] * len(sources)
    )
    for source, token_budget in zip(sources, token_budgets):
        source["token_budget"] = token_budget
        source["stats"] = {
            "source_rows_scanned": 0,
            "accepted_samples": 0,
            "accepted_tokens": 0,
        }

    with temporary_path.open("w", encoding="utf-8") as destination:
        active_sources = list(sources)
        stop_all = False
        while active_sources and not stop_all:
            made_progress = False
            for source in tuple(active_sources):
                if args.max_source_rows and source_rows_scanned >= args.max_source_rows:
                    stop_all = True
                    break
                if args.max_samples and len(token_lengths) >= args.max_samples:
                    stop_all = True
                    break

                source_budget = source["token_budget"]
                source_tokens = source["stats"]["accepted_tokens"]
                if source_budget is not None and source_tokens >= source_budget:
                    active_sources.remove(source)
                    continue

                try:
                    example = next(source["iterator"])
                except StopIteration:
                    active_sources.remove(source)
                    continue
                made_progress = True
                source_rows_scanned += 1
                source["stats"]["source_rows_scanned"] += 1

                text = extract_prompt(
                    args.workload,
                    example,
                    tokenizer,
                    use_chat_template=not args.no_chat_template,
                    rng=rng,
                    text_fields=text_fields,
                ).strip()
                if not text:
                    skipped_empty += 1
                    continue
                if not args.allow_duplicates:
                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    if digest in seen_prompts:
                        skipped_duplicate += 1
                        continue
                    seen_prompts.add(digest)

                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if len(token_ids) < args.min_document_tokens:
                    skipped_short += 1
                    continue
                if args.max_document_tokens and len(token_ids) > args.max_document_tokens:
                    if not args.truncate_long_documents:
                        skipped_long += 1
                        continue
                    token_ids = token_ids[: args.max_document_tokens]
                    text = _decode_token_ids(tokenizer, token_ids)

                budget_remaining = None
                if source_budget is not None:
                    budget_remaining = source_budget - source_tokens
                if args.token_budget:
                    global_remaining = args.token_budget - accepted_tokens
                    budget_remaining = (
                        global_remaining
                        if budget_remaining is None
                        else min(budget_remaining, global_remaining)
                    )
                if budget_remaining is not None and len(token_ids) > budget_remaining:
                    if args.truncate_long_documents and budget_remaining >= args.min_document_tokens:
                        token_ids = token_ids[:budget_remaining]
                        text = _decode_token_ids(tokenizer, token_ids)
                    else:
                        skipped_budget += 1
                        if source_budget is not None:
                            active_sources.remove(source)
                            continue
                        stop_all = True
                        break

                destination.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                token_lengths.append(len(token_ids))
                accepted_tokens += len(token_ids)
                source["stats"]["accepted_samples"] += 1
                source["stats"]["accepted_tokens"] += len(token_ids)
                if args.token_budget and accepted_tokens >= args.token_budget:
                    stop_all = True
                    break

            if not made_progress:
                break

    temporary_path.replace(output_path)
    source_manifest = []
    for source in sources:
        source_manifest.append(
            {
                "name": source["name"],
                "loader": source["loader"],
                "config": source["config"],
                "split": source["split"],
                "data_dir": source["data_dir"],
                "data_files": list(source["data_files"]),
                "token_budget": source["token_budget"],
                **source["stats"],
            }
        )
    manifest = {
        "workload": args.workload,
        "dataset": dataset_id,
        "config": config,
        "split": split,
        "tokenizer": args.tokenizer_model,
        "seed": args.seed,
        "jsonl": str(output_path),
        "requested_token_budget": args.token_budget,
        "max_samples": args.max_samples,
        "max_source_rows": args.max_source_rows,
        "min_document_tokens": args.min_document_tokens,
        "max_document_tokens": args.max_document_tokens,
        "truncate_long_documents": args.truncate_long_documents,
        "shuffle_buffer": args.shuffle_buffer,
        "deduplicate": not args.allow_duplicates,
        "balanced_sources": balance_sources,
        "sources": source_manifest,
        "source_rows_scanned": source_rows_scanned,
        "accepted_samples": len(token_lengths),
        "accepted_tokens": accepted_tokens,
        "mean_tokens": accepted_tokens / max(len(token_lengths), 1),
        "min_tokens": min(token_lengths, default=0),
        "p50_tokens": _percentile(token_lengths, 0.50),
        "p95_tokens": _percentile(token_lengths, 0.95),
        "max_tokens": max(token_lengths, default=0),
        "skipped_empty": skipped_empty,
        "skipped_short": skipped_short,
        "skipped_long": skipped_long,
        "skipped_duplicate": skipped_duplicate,
        "skipped_budget": skipped_budget,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path, manifest


def _run_megatron_preprocess(args, jsonl_path: Path) -> Path:
    if not args.megatron_dir:
        raise ValueError("--preprocess requires --megatron-dir")
    megatron_dir = Path(args.megatron_dir).expanduser().resolve()
    tool = megatron_dir / "tools" / "preprocess_data.py"
    if not tool.is_file():
        raise FileNotFoundError(f"Megatron preprocessing tool not found: {tool}")

    output_prefix = (
        Path(args.output_prefix).expanduser().resolve()
        if args.output_prefix
        else Path(
            os.environ.get("EPLB_INDEXED_DATA_DIR", args.output_dir)
        ).expanduser().resolve()
        / args.workload
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(tool),
        "--input",
        str(jsonl_path),
        "--json-keys",
        "text",
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        args.tokenizer_model,
        "--output-prefix",
        str(output_prefix),
        "--append-eod",
        "--workers",
        str(args.workers),
    ]
    print("[prepare_open_workload] running:", " ".join(command), flush=True)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(command, check=True, env=env)
    return Path(f"{output_prefix}_text_document")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-30B-A3B")
    data_root = os.environ.get(
        "EPLB_DATA_ROOT", "/mnt/hdfs/__MERLIN_USER_DIR__/eplb_data"
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("EPLB_RAW_DATA_DIR", f"{data_root}/raw"),
    )
    parser.add_argument("--output-jsonl")
    parser.add_argument("--dataset-id", help="Override the workload's Hugging Face dataset ID")
    parser.add_argument("--dataset-config", help="Override the workload's dataset config")
    parser.add_argument("--dataset-data-dir", help="Override the workload's Hugging Face data_dir")
    parser.add_argument(
        "--data-file",
        action="append",
        default=[],
        help="Explicit loader data file/URL; repeat as needed (usually with --dataset-id json)",
    )
    parser.add_argument("--text-field", help="Override the source text/content field")
    parser.add_argument("--split", help="Override the workload's dataset split")
    parser.add_argument("--max-samples", type=int, default=10_000, help="0 means unlimited")
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=100_000,
        help="Safety cap before filtering/deduplication; 0 means unlimited",
    )
    parser.add_argument("--token-budget", type=int, default=0, help="Accepted-token cap; 0 means unlimited")
    parser.add_argument("--min-document-tokens", type=int, default=16)
    parser.add_argument("--max-document-tokens", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--truncate-long-documents", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument(
        "--source-file-limit",
        type=int,
        default=1,
        help="Number of official Dolma sample shards to stream; 0 means all",
    )
    parser.add_argument(
        "--no-balance-sources",
        action="store_true",
        help="Do not split the token budget equally across multi-source workloads",
    )
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument(
        "--longbench-tasks",
        default=",".join(DEFAULT_LONGBENCH_TASKS),
        help="Comma-separated LongBench v1 task names",
    )
    parser.add_argument("--preprocess", action="store_true", help="Also create Megatron .bin/.idx")
    parser.add_argument("--megatron-dir")
    parser.add_argument("--output-prefix")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the tokenizer dependency first: pip install transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    jsonl_path, manifest = _write_workload(args, tokenizer)
    print(
        "[prepare_open_workload] "
        f"wrote {manifest['accepted_samples']} samples / {manifest['accepted_tokens']} tokens "
        f"to {jsonl_path}"
    )
    print(f"[prepare_open_workload] manifest: {jsonl_path.with_suffix('.manifest.json')}")
    if args.preprocess:
        data_path = _run_megatron_preprocess(args, jsonl_path)
        print(f"[prepare_open_workload] Megatron DATA_PATH={data_path}")


if __name__ == "__main__":
    main()
