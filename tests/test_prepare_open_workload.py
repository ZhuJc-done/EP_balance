"""Unit tests for task and pretraining-corpus workload adapters (no network access)."""

import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_open_workload.py"
SPEC = importlib.util.spec_from_file_location("prepare_open_workload", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _Tokenizer:
    chat_template = "dummy"

    @staticmethod
    def apply_chat_template(messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "|".join(f"{message['role']}:{message['content']}" for message in messages)


def test_fast_backend_decodes_truncated_tokens_without_transformers_conversion():
    class Backend:
        @staticmethod
        def decode(token_ids, skip_special_tokens):
            assert token_ids == [1, 2, 3]
            assert not skip_special_tokens
            return "decoded"

    tokenizer = SimpleNamespace(backend_tokenizer=Backend())
    assert MODULE._decode_token_ids(tokenizer, [1, 2, 3]) == "decoded"


def test_dapo_adapter_keeps_prompt_and_excludes_answer():
    text = MODULE.extract_prompt(
        "dapo_math",
        {
            "prompt": [{"role": "user", "content": "Solve 1+1."}],
            "reward_model": {"ground_truth": "2"},
        },
        _Tokenizer(),
        use_chat_template=True,
        rng=random.Random(1),
    )
    assert text == "user:Solve 1+1."
    assert "ground_truth" not in text


def test_primary_panel_uses_dapo_and_new_corpora():
    assert MODULE.CORE_WORKLOADS == (
        "dapo_math",
        "fineweb",
        "dolma",
        "fineweb2_zh",
        "starcoder",
        "pes2o",
    )


def test_pretraining_corpus_adapters_use_only_declared_text_field():
    tokenizer = _Tokenizer()
    rng = random.Random(1)
    assert MODULE.extract_prompt(
        "fineweb",
        {"text": "Web document", "answer": "ignore"},
        tokenizer,
        use_chat_template=True,
        rng=rng,
    ) == "Web document"
    assert MODULE.extract_prompt(
        "starcoder",
        {"content": "def f(): pass", "text": "wrong field"},
        tokenizer,
        use_chat_template=True,
        rng=rng,
    ) == "def f(): pass"
    assert MODULE.extract_prompt(
        "pes2o",
        {"text": "Academic paper", "abstract": "not separately appended"},
        tokenizer,
        use_chat_template=True,
        rng=rng,
    ) == "Academic paper"


def test_starcoder_sources_and_balanced_token_budgets():
    args = SimpleNamespace(
        dataset_id=None,
        dataset_data_dir=None,
        data_file=[],
        source_file_limit=1,
    )
    spec = MODULE.WORKLOADS["starcoder"]
    sources = MODULE._resolve_sources(args, spec, spec["dataset"], spec["config"], spec["split"])
    assert [source["name"] for source in sources] == ["python", "cpp", "java", "rust"]
    assert MODULE._balanced_token_budgets(10, 4) == [3, 3, 2, 2]


def test_starcoder_writer_fills_each_source_budget(tmp_path, monkeypatch):
    class Backend:
        @staticmethod
        def decode(token_ids, skip_special_tokens):
            assert not skip_special_tokens
            return "x" * len(token_ids)

    class Tokenizer:
        backend_tokenizer = Backend()

        @staticmethod
        def encode(text, add_special_tokens):
            assert not add_special_tokens
            return list(range(int(text)))

    def fake_iter(*_args, **_kwargs):
        return iter(({"content": "4"}, {"content": "4"}))

    monkeypatch.setattr(MODULE, "_iter_hf_dataset", fake_iter)
    args = SimpleNamespace(
        workload="starcoder",
        dataset_id=None,
        dataset_config=None,
        dataset_data_dir=None,
        data_file=[],
        split=None,
        text_field=None,
        seed=1,
        longbench_tasks="",
        source_file_limit=1,
        shuffle_buffer=0,
        output_dir=str(tmp_path),
        output_jsonl=None,
        max_source_rows=0,
        max_samples=0,
        token_budget=10,
        min_document_tokens=1,
        max_document_tokens=0,
        truncate_long_documents=True,
        allow_duplicates=True,
        no_chat_template=True,
        no_balance_sources=False,
        tokenizer_model="fake",
    )
    _, manifest = MODULE._write_workload(args, Tokenizer())
    assert manifest["accepted_tokens"] == 10
    assert [source["accepted_tokens"] for source in manifest["sources"]] == [3, 3, 2, 2]


def test_simple_and_longbench_adapters():
    tokenizer = _Tokenizer()
    rng = random.Random(1)
    assert MODULE.extract_prompt(
        "openscience",
        {"input": "Question", "output": "Hidden answer"},
        tokenizer,
        use_chat_template=True,
        rng=rng,
    ) == "Question"
    long_text = MODULE.extract_prompt(
        "longbench",
        {"context": "Long context", "input": "Summarize it", "answers": ["Hidden"]},
        tokenizer,
        use_chat_template=True,
        rng=rng,
    )
    assert "Long context" in long_text and "Summarize it" in long_text
    assert "Hidden" not in long_text


def test_gpqa_adapter_shuffles_all_choices_without_explanation():
    text = MODULE.extract_prompt(
        "gpqa",
        {
            "Question": "Which?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
            "Explanation": "Do not include",
        },
        _Tokenizer(),
        use_chat_template=True,
        rng=random.Random(7),
    )
    for option in ("Correct", "Wrong 1", "Wrong 2", "Wrong 3"):
        assert option in text
    assert "Do not include" not in text
