"""Задачи и их метрики: классификация, тональность, суммаризация, генерация, NER, QA."""
import json

import numpy as np
import pandas as pd
import pytest

from factory.core.contracts import MapSchemaOp, TaskSpec
from factory.prep.schema import bio_to_entities, infer_schema, to_task_frame
from factory.search.compat import check_compat
from factory.training import metrics as m
from factory.training.formatting import instruction


# ---------------------------------------------------------------- данные NER и QA

def test_bio_to_entities_handles_bio_iob1_bioes():
    toks = ["Иван", "Петров", "живёт", "в", "Нижнем", "Новгороде", "."]
    assert bio_to_entities(toks, ["B-PER", "I-PER", "O", "O", "B-LOC", "I-LOC", "O"]) == [("PER", "Иван Петров"), ("LOC", "Нижнем Новгороде")]
    assert bio_to_entities(toks, ["I-PER", "I-PER", "O", "O", "I-LOC", "I-LOC", "O"]) == [("PER", "Иван Петров"), ("LOC", "Нижнем Новгороде")]
    assert bio_to_entities(toks, ["B-PER", "E-PER", "O", "O", "B-LOC", "E-LOC", "O"]) == [("PER", "Иван Петров"), ("LOC", "Нижнем Новгороде")]
    assert bio_to_entities(["Москва", "Сочи"], ["S-LOC", "S-LOC"]) == [("LOC", "Москва"), ("LOC", "Сочи")]


def ner_frame():
    return pd.DataFrame({
        "tokens": [np.array(["Иван", "Петров", "из", "Москвы", "."]), np.array(["Просто", "текст", "."])],
        "ner_tags": [["B-PER", "I-PER", "O", "B-LOC", "O"], ["O", "O", "O"]],
        "id": [1, 2],
    })


def test_ner_schema_and_frame():
    df = ner_frame()
    schema = infer_schema(df, "entities", "generation")
    assert (schema["kind"], schema["input_column"], schema["target_column"]) == ("ner_bio", "tokens", "ner_tags")
    out = to_task_frame(df, MapSchemaOp(kind="ner_bio", input_column="tokens", target_column="ner_tags"))
    assert out["input"].tolist() == ["Иван Петров из Москвы.", "Просто текст."]
    assert out["target"].tolist() == ["PER: Иван Петров; LOC: Москвы", "нет"]


def qa_frame():
    return pd.DataFrame({
        "context": ["Пушкин родился в 1799 году в Москве. Он умер в 1837 году."],
        "question": ["Где родился Пушкин?"],
        "answers": [{"text": np.array(["в Москве", "Москве"]), "answer_start": np.array([25, 28])}],
    })


def test_qa_schema_and_frame():
    df = qa_frame()
    schema = infer_schema(df, "span", "generation")
    assert (schema["kind"], schema["input_column"], schema["target_column"], schema["question_column"]) == (
        "qa", "context", "answers", "question")
    out = to_task_frame(df, MapSchemaOp(kind="qa", input_column="context", target_column="answers", question_column="question"))
    assert out["input"][0].startswith("Контекст: Пушкин") and out["input"][0].endswith("Вопрос: Где родился Пушкин?")
    assert out["target"][0] == "в Москве" and json.loads(out["alt"][0]) == ["в Москве", "Москве"]


def test_multiline_targets_become_one_line():
    df = pd.DataFrame({"article": ["длинная статья"], "summary": ["Первая строка.\n\nВторая строка."]})
    out = to_task_frame(df, MapSchemaOp(input_column="article", target_column="summary"))
    assert out["target"][0] == "Первая строка. Вторая строка."


def _spec(registry, task_type, labels=None):
    return registry.apply_task_metrics(TaskSpec(description="d", task_type=task_type, labels=labels, base_model_id="Qwen/Qwen3-0.6B"))


def test_compat_ner_and_qa(registry):
    rep = check_compat(_spec(registry, "ner", ["PER", "LOC", "ORG"]), registry.task("ner"), ner_frame(), license_id="mit")
    assert rep.ok and rep.schema_kind == "ner_bio" and rep.label_values == ["LOC", "PER"]
    rep = check_compat(_spec(registry, "qa"), registry.task("qa"), qa_frame(), license_id="mit")
    assert rep.ok and rep.schema_kind == "qa" and rep.question_column == "question"
    # классификационный датасет для NER не подходит
    cls = pd.DataFrame({"text": ["отзыв"] * 5, "label": ["pos", "neg", "pos", "neg", "pos"]})
    assert not check_compat(_spec(registry, "ner"), registry.task("ner"), cls, license_id="mit").ok


def test_instructions_per_format(registry):
    assert "одной меткой" in instruction(_spec(registry, "text_classification", ["a", "b"]), registry.task("text_classification"))
    ner = instruction(_spec(registry, "ner", ["PER", "LOC"]), registry.task("ner"))
    assert "ТИП: текст" in ner and "PER, LOC" in ner
    assert "фрагментом из контекста" in instruction(_spec(registry, "qa"), registry.task("qa"))


# ---------------------------------------------------------------- метрики

def test_classification_metrics_include_weighted_f1():
    res = m.compute("label", ["macro_f1", "accuracy", "weighted_f1"], ["a", "a", "b", "x"], ["a", "a", "b", "b"], ["a", "b"])
    assert set(res["metrics"]) == {"macro_f1", "accuracy", "weighted_f1"}
    assert res["metrics"]["accuracy"] == 0.75 and res["unparsed_fraction"] == 0.25


def test_summarization_rouge_family():
    res = m.compute("text", ["rougeL", "rouge1", "rouge2", "chrf"], ["кошка сидит на окне"], ["кошка сидит на окне"], None)
    assert res["metrics"]["rouge1"] == pytest.approx(1.0) and res["metrics"]["rouge2"] == pytest.approx(1.0)
    assert res["metrics"]["rougeL"] == pytest.approx(1.0)


def test_ner_entity_level_metrics():
    refs = ["PER: Иван Петров; LOC: Москва", "нет", "ORG: Яндекс"]
    preds = ["PER: иван петров; LOC: Казань", "нет", "ORG: Яндекс; PER: Лишний"]
    res = m.compute("entities", ["entity_f1", "entity_precision", "entity_recall"], preds, refs, None)
    # tp: Иван Петров, Яндекс = 2; fp: Казань, Лишний = 2; fn: Москва = 1
    assert res["metrics"]["entity_precision"] == pytest.approx(0.5)
    assert res["metrics"]["entity_recall"] == pytest.approx(2 / 3)
    assert res["metrics"]["entity_f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert res["per_class"]["LOC"]["recall"] == 0.0 and res["per_class"]["PER"]["precision"] == 0.5


def test_ner_unparsed_answers_counted():
    res = m.compute("entities", ["entity_f1"], ["какой-то бред без формата"], ["PER: Иван"], None)
    assert res["metrics"]["entity_f1"] == 0.0 and res["unparsed_fraction"] == 1.0


def test_qa_squad_metrics_take_best_alternative():
    res = m.compute("span", ["token_f1", "exact_match"], ["Москве", "в 1837"], ["в Москве", "в 1799 году"], None,
                    alternatives=[["в Москве", "Москве"], ["в 1799 году"]])
    assert res["metrics"]["exact_match"] == pytest.approx(0.5)  # «Москве» совпало со вторым эталоном
    assert res["metrics"]["token_f1"] == pytest.approx((1.0 + 2 * 0.5 * (1 / 3) / (0.5 + 1 / 3)) / 2)


def test_example_correctness_per_format():
    assert m.example_correct("entities", "LOC: москва; PER: Иван", "", "PER: Иван; LOC: Москва", None)
    assert m.example_correct("span", "москве!", "", "в Москве", ["в Москве", "Москве"])
    assert not m.example_correct("text", "да", "", "нет", None)


# ---------------------------------------------------------------- baseline'ы

def test_gazetteer_baseline():
    preds = m.simple_baseline_predictions("gazetteer", ["Вчера Иван Петров прилетел в Москву и Москва встретила"],
                                          ["PER: Иван Петров", "LOC: Москва", "LOC: Москву"], [])
    assert preds == ["PER: Иван Петров; LOC: Москву; LOC: Москва"]
    assert m.simple_baseline_predictions("gazetteer", ["ничего"], ["PER: Иван"], []) == ["нет"]


def test_best_sentence_baseline():
    inp = "Контекст: Пушкин родился в Москве. Он писал стихи. Поэт умер в Петербурге.\nВопрос: Где умер поэт?"
    assert m.simple_baseline_predictions("best_sentence", [inp], [], []) == ["Поэт умер в Петербурге."]
    inp = "Контекст: Музей стоит в Казани. Театр находится в Москве.\nВопрос: Где находится театр в Москва?"
    assert m.simple_baseline_predictions("best_sentence", [inp], [], []) == ["Театр находится в Москве."]


def test_every_task_has_working_metrics_and_baselines(registry):
    """Каждая поддерживаемая задача: все метрики реестра считаются, baseline'ы реестра существуют."""
    samples = {"label": ("a", "a"), "text": ("кот спит", "кот спит"), "entities": ("PER: Иван", "PER: Иван"),
               "span": ("Москве", "Москве")}
    for task in registry.supported_tasks():
        pred, ref = samples[task.format]
        names = [n for n in task.metrics if n != "bert_score_f1"]
        res = m.compute(task.format, names, [pred], [ref], ["a", "b"])
        assert set(res["metrics"]) == set(names), task.key
        inputs = ["Контекст: Москва большая.\nВопрос: Какая Москва?"] if task.format == "span" else ["Иван пришёл"]
        for b in task.baselines:
            if b != "base_prompt":
                assert m.simple_baseline_predictions(b, inputs, [ref], [len(ref)]) is not None, (task.key, b)
