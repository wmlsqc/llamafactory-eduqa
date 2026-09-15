import json
from pathlib import Path

import pytest

from eduqa.data import load_jsonl, prepare_data, prompt_key, sha256_file


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def row(question: str, answer: str = "解释\n第二行") -> dict:
    return {"instruction": question, "input": "", "output": answer}


def test_reproducible_splits_and_hashed_manifest(tmp_path):
    source = write_jsonl(tmp_path / "source" / "train.jsonl", [row(f"问题{i}") for i in range(20)])
    test = write_jsonl(tmp_path / "source" / "test.jsonl", [row("独立问题")])
    first = prepare_data(source, test, tmp_path / "first", val_ratio=0.2, seed=71)
    second = prepare_data(source, test, tmp_path / "second", val_ratio=0.2, seed=71)
    assert first == second
    assert first["counts"] == {"train": 16, "val": 4, "test": 1}
    splits = {}
    for name in ("train", "val", "test"):
        path = tmp_path / "first" / f"{name}.json"
        assert sha256_file(path) == first["files"][path.name]["sha256"]
        assert path.read_bytes() == (tmp_path / "second" / path.name).read_bytes()
        splits[name] = {prompt_key(item) for item in json.loads(path.read_text(encoding="utf-8"))}
    assert not splits["train"] & splits["val"]
    assert not (splits["train"] | splits["val"]) & splits["test"]
    info = json.loads((tmp_path / "first" / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["eduqa_train"]["file_name"] == "train.json"
    assert info["eduqa_val"]["columns"]["response"] == "output"


def test_normalization_deduplication_and_blank_line_accounting(tmp_path):
    source = write_jsonl(tmp_path / "source.jsonl", [row("  Ａ  B  "), row("a\tb")])
    with source.open("a", encoding="utf-8") as stream:
        stream.write("\n  \n")
    rows, counts = load_jsonl(source)
    assert len(rows) == 1
    assert rows[0]["instruction"] == "A  B"
    assert rows[0]["output"] == "解释\n第二行"
    assert counts == {"source_records": 2, "blank_lines": 2, "duplicates_removed": 1, "unique_records": 1}


def test_same_prompt_conflicting_answers_is_error(tmp_path):
    source = write_jsonl(tmp_path / "source.jsonl", [row("问题", "答案一"), row("问题", "答案二")])
    with pytest.raises(ValueError, match="conflicting answers.*line 1"):
        load_jsonl(source)


@pytest.mark.parametrize("payload,match", [
    ("{invalid}\n", "invalid JSON"),
    ('[]\n', "must be an object"),
    ('{"instruction":"问题","output":null}\n', "output must be a string"),
    ('{"instruction":"问题","output":" "}\n', "output must not be empty"),
    ('{"instruction":17,"output":"答案"}\n', "instruction must be a string"),
    ('{"instruction":"问题","input":{},"output":"答案"}\n', "input must be a string"),
    ('{"instruction":"问题","output":"答案","history":[]}\n', "unsupported fields"),
    ("\n", "no valid records"),
])
def test_invalid_data_never_silently_disappears(tmp_path, payload, match):
    source = tmp_path / "bad.jsonl"
    source.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_jsonl(source)


def test_normalized_leakage_blocks_output(tmp_path):
    source = write_jsonl(tmp_path / "source" / "train.jsonl", [row("Ｆ=ｍａ"), row("第二个问题")])
    test = write_jsonl(tmp_path / "source" / "test.jsonl", [row(" f=ma ", "其它答案")])
    with pytest.raises(ValueError, match="Train/test leakage"):
        prepare_data(source, test, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_input_participates_in_prompt_identity(tmp_path):
    train_rows = [row("解题"), row("另一题")]
    train_rows[0]["input"] = "1+1"
    test_row = row("解题")
    test_row["input"] = "2+2"
    source = write_jsonl(tmp_path / "source" / "train.jsonl", train_rows)
    test = write_jsonl(tmp_path / "source" / "test.jsonl", [test_row])
    assert prepare_data(source, test, tmp_path / "out")["counts"] == {"train": 1, "val": 1, "test": 1}


def test_overwrite_requires_flag_and_invalid_source_preserves_previous_output(tmp_path):
    source = write_jsonl(tmp_path / "source" / "train.jsonl", [row("一"), row("二")])
    test = write_jsonl(tmp_path / "source" / "test.jsonl", [row("三")])
    output = tmp_path / "out"
    prepare_data(source, test, output)
    previous = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_data(source, test, output)
    source.write_text("bad data", encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_data(source, test, output, overwrite=True)
    assert (output / "manifest.json").read_bytes() == previous


@pytest.mark.parametrize("ratio", [0, 1, -1, float("nan"), float("inf")])
def test_invalid_validation_ratio(tmp_path, ratio):
    with pytest.raises(ValueError, match="val_ratio"):
        prepare_data(tmp_path / "missing", tmp_path / "missing", tmp_path / "out", val_ratio=ratio)
