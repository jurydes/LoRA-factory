"""Ядро: хранилище, автомат состояний, режимы, контракты, реестры, поиск, раунды."""

import pandas as pd
import pytest

from factory.core.contracts import Constraints, MonitorDecision, TaskSpec
from factory.core.state import SessionState, Step, TransitionError, is_blocking
from factory.core.storage import append_jsonl, atomic_write_text, file_lock, read_jsonl
from factory.search.catalog import Catalog
from factory.search.compat import check_compat, check_license
from factory.search.llm_eval import CRITERIA, DatasetAssessment, weighted_score
from factory.search.retrieval import bm25_ranking, rrf
from factory.training import metrics, rounds


def test_atomic_write_and_jsonl_offsets(tmp_path):
    p = tmp_path / "a.json"
    atomic_write_text(p, "1")
    atomic_write_text(p, "2")
    assert p.read_text() == "2" and not list(tmp_path.glob(".a.json.*"))
    j = tmp_path / "m.jsonl"
    append_jsonl(j, {"a": 1})
    recs, off = read_jsonl(j)
    with open(j, "a") as f:
        f.write('{"a": 2}\n{"a": 3')  # последняя строка ещё пишется
    recs2, off2 = read_jsonl(j, off)
    assert recs == [{"a": 1}] and recs2 == [{"a": 2}]
    with open(j, "a") as f:
        f.write("}\n")
    assert read_jsonl(j, off2)[0] == [{"a": 3}]


def test_file_lock_is_exclusive(tmp_path):
    with file_lock(tmp_path / "gpu0.lock") as got:
        assert got
        with file_lock(tmp_path / "gpu0.lock", blocking=False) as again:
            # flock на разных файловых дескрипторах одного процесса конфликтует
            assert again is False


def test_transitions():
    st = SessionState(session_id="s")
    st.goto(Step.SPEC_CONFIRM)
    st.goto(Step.SEARCH)
    with pytest.raises(TransitionError):
        st.goto(Step.TRAINING)
    st.goto(Step.FAILED)  # из любого
    st.goto(Step.INTAKE)


def test_is_blocking_modes():
    for card in ("spec", "dataset_choice", "prep", "config", "round", "augment"):
        assert is_blocking(card, "expert")
    assert not is_blocking("spec", "auto")
    assert is_blocking("config", "auto")  # итоговая конфигурация — точка необратимости
    assert not is_blocking("prep", "auto", {"mass_removal": False})
    assert is_blocking("prep", "auto", {"mass_removal": True})
    assert not is_blocking("augment", "auto", {"confident": True})
    assert not is_blocking("round", "auto", {"sufficient": True})  # §3.4: достаточно — завершает система
    assert is_blocking("round", "auto", {"sufficient": False})     # иначе спрашивает пользователя
    assert not is_blocking("result", "expert")


def test_registry_metrics_come_from_registry(registry):
    spec = TaskSpec(description="d", task_type="sentiment_analysis", labels=["a", "b"], base_model_id="Qwen/Qwen3-0.6B",
                    primary_metric="bleu", metrics=["придумано LLM"])
    out = registry.apply_task_metrics(spec)
    assert out.primary_metric == "macro_f1" and out.metrics == ["macro_f1", "accuracy", "weighted_f1"]
    with pytest.raises(ValueError):
        registry.apply_task_metrics(spec.model_copy(update={"labels": None}))
    with pytest.raises(KeyError):
        registry.apply_task_metrics(spec.model_copy(update={"task_type": "unknown"}))
    ner = registry.apply_task_metrics(spec.model_copy(update={"task_type": "ner", "labels": None}))
    assert ner.primary_metric == "entity_f1"  # NER: типы сущностей необязательны — возьмутся из данных


def test_model_filter(registry):
    small = registry.filter_models(Constraints(max_model_params_b=1.0), "ru")
    assert "Qwen/Qwen3-0.6B" in [m.id for m in small] and all(m.params_b <= 1.0 for m in small)
    assert all(m.commercial_ok for m in registry.filter_models(Constraints(commercial_license=True)))


def test_monitor_decision_patch_consistency():
    with pytest.raises(ValueError):
        MonitorDecision(verdict="ok", action="patch", explanation="x")
    with pytest.raises(ValueError):
        MonitorDecision.model_validate({"verdict": "ok", "action": "patch", "explanation": "x",
                                        "patch": {"kind": "lr_scale", "factor": 2.0}})  # только уменьшение


def test_rrf():
    s = rrf([[1, 2, 3], [3, 1]], k=60)
    assert s[1] == pytest.approx(1 / 61 + 1 / 62)
    assert max(s, key=s.get) == 1


def test_bm25_ranking():
    # BM25Okapi: idf слова, встречающегося в половине документов, равен нулю — корпус из нескольких документов
    cat = Catalog(records=[{"id": "a/toxic", "description": "детоксификация токсичных комментариев", "tags": []},
                           {"id": "b/news", "description": "новости и заголовки", "tags": []},
                           {"id": "c/qa", "description": "вопросы и ответы по википедии", "tags": []},
                           {"id": "d/sum", "description": "суммаризация статей", "tags": []}], embeddings=None)
    assert bm25_ranking(cat, "токсичных комментариев")[0] == 0


def test_weighted_score():
    full = DatasetAssessment(**{k: {"score": 5, "reason": "r"} for k in CRITERIA}, why="w")
    assert weighted_score(full, {k: 1.0 for k in CRITERIA}) == 100.0
    half = DatasetAssessment(**{k: {"score": 0 if k == "relevance" else 5, "reason": "r"} for k in CRITERIA}, why="w")
    assert weighted_score(half, {"relevance": 1.0, "size": 1.0}) == 50.0


def test_compat(cls_spec, registry):
    task = registry.task(cls_spec.task_type)
    sample = pd.DataFrame({"text": ["отзыв номер %d довольно длинный" % i for i in range(20)],
                           "label": ["позитив", "негатив"] * 10})
    rep = check_compat(cls_spec, task, sample, license_id="mit")
    assert rep.ok and rep.input_column == "text" and rep.label_overlap == pytest.approx(2 / 3)
    assert not check_compat(cls_spec, task, sample, has_loading_script=True).ok
    assert check_license(None, commercial=True)[0] is False
    assert check_license("apache-2.0", commercial=True)[0] is True


def test_label_parsing_and_metrics():
    labels = ["позитив", "негатив"]
    assert metrics.parse_label(" Позитив.", labels) == "позитив"
    assert metrics.parse_label("скорее негатив", labels) == "негатив"
    assert metrics.parse_label("не знаю", labels) is None
    m, per_class, unparsed, _ = metrics.classification_metrics(["позитив", "мусор"], ["позитив", "негатив"], labels)
    assert m["accuracy"] == 0.5 and unparsed == 0.5
    g = metrics.generation_metrics(["привет мир"], ["привет мир"], ["chrf", "exact_match", "rougeL"])
    assert g["exact_match"] == 1.0 and g["rougeL"] == pytest.approx(1.0)


def test_sufficiency():
    base = {"majority": {"macro_f1": 0.3}}
    assert rounds.is_sufficient("macro_f1", {"macro_f1": 0.8}, base, 0.7, None, 0.005)[0]
    assert not rounds.is_sufficient("macro_f1", {"macro_f1": 0.6}, base, 0.7, None, 0.005)[0]
    assert not rounds.is_sufficient("macro_f1", {"macro_f1": 0.25}, base, None, 0.2, 0.005)[0]
    assert rounds.is_sufficient("macro_f1", {"macro_f1": 0.6}, base, None, 0.598, 0.005)[0]
    assert not rounds.is_sufficient("macro_f1", {"macro_f1": 0.6}, base, None, 0.5, 0.005)[0]


def test_placeholders():
    values = rounds.flatten_values(val={"macro_f1": 0.81234}, baseline={"majority": {"macro_f1": 0.3}})
    assert rounds.check_placeholders("F1 {val.macro_f1} против {baseline.majority.macro_f1}, 2 раунда", values) == []
    assert rounds.check_placeholders("F1 0.81", values)
    assert rounds.check_placeholders("F1 81%", values)
    assert rounds.check_placeholders("{val.unknown}", values)
    assert rounds.render("F1 {val.macro_f1}", values) == "F1 0.8123"


def test_label_map_guard():
    from factory.search.compat import auto_label_map, checked_label_map
    assert auto_label_map(["Позитив", "neg"], ["позитив", "негатив"]) == {"Позитив": "позитив"}
    proposed = {"0": "негатив", "1": "позитив", "7": "позитив", "2": "выдумано"}
    assert checked_label_map(proposed, ["0", "1", "2"], ["позитив", "негатив"]) == {"0": "негатив", "1": "позитив"}
