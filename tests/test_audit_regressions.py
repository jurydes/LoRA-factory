"""Пограничные сценарии данных, загрузки и восстановления после сбоя."""
import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from factory.core.contracts import SplitProportions
from factory.core.storage import append_jsonl, write_json
from factory.core.state import Step
from factory.orchestrator import Orchestrator
from factory.prep.ops import normalize_text
from factory.prep.split import split_frame
from factory.runs import RunPaths
from factory.training import metrics
from factory.training.checkpoints import last_checkpoint, new_attempt


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_conflicting_labels_do_not_split_identical_inputs(seed):
    rows = [(f"text {i}", label) for label in ["a", "b"] for i in range(30)]
    rows += [("  TEXT 0  ", "c")]
    df = pd.DataFrame(rows, columns=["input", "target"])
    splits = split_frame(df, SplitProportions(), seed, stratify=True)
    groups = [set(s["input"].map(normalize_text)) for s in splits]
    assert all(len(s) for s in splits)
    assert sum(map(len, splits)) == len(df)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    for a, b in zip(splits, split_frame(df, SplitProportions(), seed, stratify=True)):
        pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("prediction,expected", [
    ("positively wrong", None), ("positive: good", "positive"),
    ("The label is negative.", "negative"), ("negative", "negative"),
    ("unpositive", None), ("", None),
])
def test_label_requires_whole_word(prediction, expected):
    assert metrics.parse_label(prediction, ["positive", "negative", "!!!"]) == expected


def test_empty_generation_has_zero_similarity(monkeypatch):
    monkeypatch.setattr(metrics, "_bert_score_f1", lambda *a: pytest.fail("empty responses need no BERT model"))
    names = ["chrf", "bleu", "rouge1", "rouge2", "rougeL", "exact_match", "bert_score_f1"]
    scores = metrics.generation_metrics(["", " \n"], ["нет", "нет"], names)
    assert scores == dict.fromkeys(names, 0.0)


def test_bert_score_counts_empty_predictions_in_denominator(monkeypatch):
    def score(preds, refs):
        assert preds == ["ответ"] and refs == ["ответ"]
        return 1.0
    monkeypatch.setattr(metrics, "_bert_score_f1", score)
    assert metrics.generation_metrics(["", "ответ"], ["нет", "ответ"], ["bert_score_f1"]) == {"bert_score_f1": 0.5}


@pytest.mark.parametrize("excluded", [False, True])
def test_empty_dataset_search_returns_no_candidates(cfg, registry, cls_spec, monkeypatch, excluded):
    from factory.search import funnel
    class Catalog:
        records = [{"id": "excluded"}]
        def __len__(self):
            return len(self.records)
    monkeypatch.setattr(funnel, "expand_query", AsyncMock(return_value="query"))
    monkeypatch.setattr(funnel, "load_catalog", lambda *a: Catalog())
    monkeypatch.setattr(funnel, "hybrid_search", lambda *a: [(0, {})] if excluded else [])
    monkeypatch.setattr(funnel, "ProcessPoolExecutor", lambda **kw: pytest.fail("empty search needs no workers"))
    assert asyncio.run(funnel.search(cls_spec, registry, cfg, exclude={"excluded"})) == []


def test_dataset_search_timeout_stops_workers_before_pool_shutdown(cfg, registry, cls_spec, monkeypatch):
    from factory.search import funnel

    class Catalog:
        records = [{"id": "slow", "description": "slow"}]

        def __len__(self):
            return 1

    class Process:
        alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.alive = False

        def join(self, timeout):
            pass

    class Pool:
        def __init__(self):
            self.process = Process()
            self._processes = {1: self.process}
            self.closed = False

        def submit(self, *args):
            return Future()  # загрузка HF зависла и никогда не возвращает результат

        def shutdown(self, wait, cancel_futures):
            assert not self.process.is_alive()
            self.closed = True

    pool = Pool()
    monkeypatch.setattr(funnel, "expand_query", AsyncMock(return_value="query"))
    monkeypatch.setattr(funnel, "load_catalog", lambda *a: Catalog())
    monkeypatch.setattr(funnel, "hybrid_search", lambda *a: [(0, {})])
    monkeypatch.setattr(funnel, "ProcessPoolExecutor", lambda **kw: pool)
    short = cfg.model_copy(update={"search": cfg.search.model_copy(update={"compat_timeout_s": 0.02})})
    progress = []
    assert asyncio.run(funnel.search(cls_spec, registry, short, progress=progress.append)) == []
    assert pool.closed
    assert any("Проверено" in message for message in progress)


@pytest.mark.parametrize("step,busy", [(Step.PREP, None), (Step.INTAKE, "dialogue"), (Step.DATASET_CHOICE, "prep")])
def test_rejected_upload_preserves_file_and_state(cfg, registry, step, busy):
    orch = Orchestrator(cfg, registry)
    st = orch.create_session()
    st.step, st.busy = step, busy
    orch.save(st)
    path = orch.session_dir(st.session_id) / "uploads" / "data.csv"
    path.parent.mkdir()
    path.write_bytes(b"original dataset")
    before = orch.load(st.session_id)
    with pytest.raises((ValueError, RuntimeError)):
        asyncio.run(orch.on_upload(st.session_id, "data.csv", b"replaced dataset"))
    assert path.read_bytes() == b"original dataset"
    assert orch.load(st.session_id) == before


def test_accepted_upload_is_saved(cfg, registry):
    orch = Orchestrator(cfg, registry)
    st = orch.create_session()
    result = asyncio.run(orch.on_upload(st.session_id, "data.csv", b"new dataset"))
    from pathlib import Path
    assert Path(result.user_dataset_path).read_bytes() == b"new dataset"


def test_reattach_preserves_offsets(cfg, registry, monkeypatch, tmp_path):
    orch = Orchestrator(cfg, registry)
    st = orch.create_session()
    st.step = Step.TRAINING
    orch.save(st)
    paths = RunPaths(tmp_path / "run")
    append_jsonl(paths.events, {"type": "round_started", "epochs": 2, "step_offset": 500, "epoch_offset": 3})
    captured = []
    async def train(sid, plan, config, attach_pid=None):
        captured.append(plan)
    async def background(sid, label, fn):
        await fn()
    monkeypatch.setattr(orch, "_run_training", train)
    monkeypatch.setattr(orch, "_background", background)
    manifest = SimpleNamespace(session_id=st.session_id, rounds=[], config=SimpleNamespace(schedule=SimpleNamespace(epochs=2)), pid=123)
    asyncio.run(orch._reattach(manifest, paths))
    assert captured[0].step_offset == 500 and captured[0].epoch_offset == 3


def test_latest_attempt_wins_over_larger_old_checkpoint(tmp_path):
    legacy = tmp_path / "checkpoint-900"
    write_json(legacy / "trainer_state.json", {"global_step": 900})
    assert last_checkpoint(tmp_path) == legacy
    first = new_attempt(tmp_path)
    old = first / "checkpoint-100"
    write_json(old / "trainer_state.json", {"global_step": 100})
    second = new_attempt(tmp_path)
    assert last_checkpoint(second) is None
    assert last_checkpoint(tmp_path) == old
    latest = second / "checkpoint-20"
    write_json(latest / "trainer_state.json", {"global_step": 20})
    (second / "checkpoint-30").mkdir()  # сбой во время сохранения
    assert last_checkpoint(tmp_path) == latest
    assert (old / "trainer_state.json").exists()


def test_legacy_checkpoint_selection_uses_save_time(tmp_path):
    import os
    old, new = tmp_path / "checkpoint-100", tmp_path / "checkpoint-20"
    write_json(old / "trainer_state.json", {"global_step": 100})
    write_json(new / "trainer_state.json", {"global_step": 20})
    os.utime(old / "trainer_state.json", (100, 100))
    os.utime(new / "trainer_state.json", (200, 200))
    assert last_checkpoint(tmp_path) == new
