"""Блок 2: операции рецепта, разбиение с закрытым test, профиль, статистические правила."""
import pandas as pd
import pytest

from factory.core.config import PrepRules
from factory.core.contracts import (
    DataProfile, DedupeOp, DropLabelNoiseOp, FilterOp, LengthStats, MapSchemaOp, RebalanceOp, Recipe, SplitProportions,
)
from factory.prep import ops, rules
from factory.prep.schema import infer_schema, to_task_frame
from factory.prep.profile import profile_frame
from factory.prep.split import SealedTestError, read_test, save_splits, split_frame


def frame(rows):
    df = pd.DataFrame(rows, columns=["input", "target"])
    df["synthetic"] = False
    df.insert(0, "_id", [f"id{i}" for i in range(len(df))])
    return df


@pytest.fixture
def cls_df():
    rows = [(f"отличный товар номер {i}, всем советую", "позитив") for i in range(60)]
    rows += [(f"ужасное качество, вещь {i} сломалась", "негатив") for i in range(20)]
    rows += [("отличный товар номер 1, всем советую", "позитив")] * 3  # точные дубли
    rows += [("", "позитив")]
    return frame(rows)


def test_exact_dedupe(cls_df):
    out = ops.apply_dedupe(cls_df, DedupeOp(mode="exact"))
    assert len(out) == len(cls_df) - 3


def test_near_dedupe():
    df = frame([("мама мыла раму очень долго и тщательно вчера вечером", "a"),
                ("мама мыла раму очень долго и тщательно вчера вечером!", "a"),
                ("совершенно другой текст про погоду и прогулки в парке", "b")])
    assert len(ops.apply_dedupe(df, DedupeOp(mode="near", threshold=0.8))) == 2


def test_filter_closed_fields(cls_df):
    out = ops.apply_filter(cls_df, FilterOp(field="input_len", operator=">", value=0))
    assert (out["input"] != "").all()
    out = ops.apply_filter(cls_df, FilterOp(field="label", operator="in", value=["негатив"]))
    assert set(out["target"]) == {"негатив"}
    with pytest.raises(Exception):
        FilterOp(field="custom_code", operator=">", value=0)


def test_rebalance(cls_df):
    under = ops.apply_rebalance(cls_df, RebalanceOp(method="undersample", max_ratio=1.0))
    assert under["target"].value_counts().nunique() == 1
    over = ops.apply_rebalance(cls_df, RebalanceOp(method="oversample", max_ratio=1.0))
    vc = over["target"].value_counts()
    assert vc.max() == vc.min()


def test_drop_label_noise_capped():
    rows = [(f"хороший отличный прекрасный {i}", "pos") for i in range(100)]
    rows += [(f"плохой ужасный отвратительный {i}", "neg") for i in range(100)]
    rows += [("плохой ужасный отвратительный", "pos")] * 4  # явный шум меток
    df = frame(rows)
    out = ops.apply_drop_label_noise(df, DropLabelNoiseOp(max_fraction=0.05), seed=0)
    assert len(df) - len(out) <= int(len(df) * 0.05)
    assert len(out) < len(df)


def test_recipe_application_is_deterministic(cls_df):
    r = Recipe(dataset_id="x", split_seed=7, ops=[
        FilterOp(field="input_len", operator=">", value=0), DedupeOp(mode="exact"),
        RebalanceOp(method="undersample", max_ratio=2.0)])
    a, b = ops.apply_recipe(cls_df, r), ops.apply_recipe(cls_df, r)
    pd.testing.assert_frame_equal(a, b)


def test_split_groups_duplicates_and_stratifies(cls_df):
    train, val, test = split_frame(cls_df, SplitProportions(), seed=1, stratify=True)
    assert len(train) + len(val) + len(test) == len(cls_df)
    norm = lambda df: set(df["input"].map(ops.normalize_text))
    assert not (norm(train) & norm(test)) and not (norm(val) & norm(test))
    assert set(test["target"]) == {"позитив", "негатив"}


def test_test_is_sealed(tmp_path, cls_df):
    train, val, test = split_frame(cls_df, SplitProportions(), seed=1, stratify=True)
    h = save_splits(tmp_path, train, val, test)
    with pytest.raises(SealedTestError):
        read_test(tmp_path, final=False, expected_hash=h)
    assert len(read_test(tmp_path, final=True, expected_hash=h)) == len(test)
    with pytest.raises(SealedTestError):
        read_test(tmp_path, final=True, expected_hash="другой")


def test_profile(cls_df):
    p = profile_frame(cls_df, "classification")
    assert p.n_rows == len(cls_df) and p.n_exact_dupes == 3 and p.n_empty == 1
    assert p.imbalance_ratio == pytest.approx(64 / 20)


def _profile(**kw):
    base = dict(n_rows=5000, language_share={"ru": 0.99}, label_counts={"a": 2500, "b": 2500}, imbalance_ratio=1.0)
    return DataProfile(**(base | kw))


def test_rule_clean_data_needs_no_agent():
    v = rules.assess(_profile(), "classification", PrepRules(), "ru")
    assert not v.call_agent and not v.ops


def test_rule_empty_and_dupes_fixed_by_rules():
    v = rules.assess(_profile(n_empty=10, n_exact_dupes=20), "classification", PrepRules(), "ru")
    assert {o.op for o in v.ops} == {"filter", "dedupe"} and not v.call_agent


def test_rule_many_dupes_calls_agent():
    assert rules.assess(_profile(n_exact_dupes=500), "classification", PrepRules(), "ru").call_agent


def test_rule_imbalance_calls_agent():
    assert rules.assess(_profile(imbalance_ratio=25.0), "classification", PrepRules(), "ru").call_agent


def test_rule_language_mix_calls_agent():
    assert rules.assess(_profile(language_share={"ru": 0.5, "en": 0.5}), "classification", PrepRules(), "ru").call_agent


def test_rule_long_targets_call_agent():
    p = _profile(target_tokens_est=LengthStats(p99=5000))
    assert rules.assess(p, "generation", PrepRules(), "ru").call_agent


def test_rule_small_dataset_calls_agent():
    assert rules.assess(_profile(n_rows=100), "classification", PrepRules(), "ru").call_agent


def test_infer_columns_and_map_schema():
    raw = pd.DataFrame({"text": ["длинный текст отзыва номер %d" % i for i in range(30)],
                        "label": ["pos", "neg", "neu"] * 10, "idx": range(30)})
    schema = infer_schema(raw, "label", "classification")
    i, t = schema["input_column"], schema["target_column"]
    assert (i, t) == ("text", "label")
    df = to_task_frame(raw, MapSchemaOp(input_column=i, target_column=t, label_map={"pos": "позитив"}))
    assert list(df.columns) == ["_id", "input", "target", "synthetic"] and "позитив" in set(df["target"])
    gen = pd.DataFrame({"toxic_comment": ["злой текст %d" % i for i in range(10)],
                        "neutral_comment": ["добрый текст %d" % i for i in range(10)]})
    schema = infer_schema(gen, "text", "generation")
    assert (schema["input_column"], schema["target_column"]) == ("toxic_comment", "neutral_comment")


def test_quality_report_goals(cfg):
    from factory.prep.quality import failing, quality_report
    rows = [(f"нормальный текст отзыва номер {i} про еду и сервис", "pos" if i % 3 else "neg") for i in range(60)]
    rows += [("нормальный текст отзыва номер 1 про еду и сервис", "pos")] * 2       # точные дубли
    rows += [("", "pos"), ("<div>купить</div> http://spam.example", "neg")]         # пустой и мусор
    rows += [("ок", "unknown")]                                                      # метка вне постановки
    goals = quality_report(frame(rows), "classification", cfg.dataprep.rules, "ru", ["pos", "neg"], label_noise=False)
    bad = set(failing(goals))
    assert {"no_empty_rows", "no_exact_duplicates", "junk_text", "labels_in_spec"} <= bad
    assert goals["no_exact_duplicates"]["value"] == 2


def test_clean_text_and_map_labels_ops():
    from factory.core.contracts import CleanTextOp, MapLabelsOp
    df = frame([("<b>Супер</b>!!!!!!!   http://x.ru  &amp; ок", "positive"), ("плохо", "neg")])
    out = ops.apply_clean_text(df, CleanTextOp(remove_urls=True, max_char_repeat=3))
    assert out["input"][0] == "Супер !!! & ок"
    out = ops.apply_map_labels(df, MapLabelsOp(mapping={"positive": "pos"}))
    assert out["target"].tolist() == ["pos", "neg"]


def test_synonym_perturbation_is_deterministic():
    import random
    from factory.prep.augment import _perturb
    syn = {"хороший": ["отличный"], "фильм": ["кинофильм"]}
    a = [_perturb("Очень хороший фильм, всем советую посмотреть", random.Random(s), syn) for s in range(20)]
    b = [_perturb("Очень хороший фильм, всем советую посмотреть", random.Random(s), syn) for s in range(20)]
    assert a == b and any("отличный" in t or "кинофильм" in t for t in a)
