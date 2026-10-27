"""Сопоставление колонок для датасетов разной структуры. Таблицы повторяют форму реальных датасетов HF
(alpaca, OpenOrca, ultrachat, opus, m_arc, belebele, ARC, COPA, go_emotions, kinopoisk, MNLI, wmt24pp, multinerd,
pii-bench) — проверка без сети."""
import json

import numpy as np
import pandas as pd
import pytest

from factory.core.contracts import TaskSpec
from factory.prep.loading import _decode_class_labels, choose_splits, pick_config
from factory.prep.schema import bio_to_entities, drop_unusable, infer_schema, to_task_frame
from factory.search.compat import check_compat, schema_op
from factory.search.llm_eval import MappingFix, apply_mapping_fix, build_prompt


def spec(registry, task_type, labels=None, description="задача"):
    return registry.apply_task_metrics(TaskSpec(description=description, task_type=task_type, labels=labels,
                                                base_model_id="Qwen/Qwen3-0.6B"))


def compat(registry, df, task_type, labels=None, description="задача", label_names=None):
    return check_compat(spec(registry, task_type, labels, description), registry.task(task_type), df,
                        label_names, license_id="mit")


def frame(rep, df, fmt=None):
    return drop_unusable(to_task_frame(df, schema_op(rep), fmt))[0]


RU = ["Объясни, почему небо голубое", "Назови столицу Франции", "Переведи слово «кошка»", "Сложи два и три",
      "Придумай название для кафе", "Опиши осень одним предложением"]


# ---------------------------------------------------------------- генерация

def test_alpaca_instruction_and_input_without_leaking_answer(registry):
    df = pd.DataFrame({
        "instruction": RU * 5,
        "input": ["", "", "кошка", "", "у моря", ""] * 5,
        "output": [f"Ответ номер {i} на задание, довольно подробный" for i in range(30)],
    })
    df["text"] = "### Инструкция: " + df["instruction"] + " ### Ответ: " + df["output"]  # полный пример с ответом
    rep = compat(registry, df, "text_generation")
    assert rep.ok and (rep.input_column, rep.extra_input_columns, rep.target_column) == ("instruction", ["input"], "output")
    out = frame(rep, df)
    assert out["input"][2] == "Инструкция: Переведи слово «кошка»\nТекст: кошка"
    assert out["input"][0] == "Инструкция: Объясни, почему небо голубое"  # пустой input пропущен
    assert not any("### Ответ" in i for i in out["input"])


def test_system_prompt_and_question(registry):
    df = pd.DataFrame({"id": [f"niv.{i}" for i in range(30)],
                       "system_prompt": ["Вы помощник ИИ. Отвечайте подробно.", "Вы полезный ассистент."] * 15,
                       "question": [f"{q} (вариант {i})" for i, q in enumerate(RU * 5)],
                       "response": [f"Развёрнутый ответ {i}" for i in range(30)]})
    rep = compat(registry, df, "text_generation")
    assert [rep.input_column, *rep.extra_input_columns] == ["system_prompt", "question"] and rep.target_column == "response"
    assert frame(rep, df)["input"][0].startswith("Системная инструкция: Вы помощник ИИ. Отвечайте подробно.\nВопрос: Объясни")


@pytest.mark.parametrize("serialized", [False, True])
def test_chat_messages_first_exchange(registry, serialized):
    def dialog(i):
        msgs = [{"role": "system", "content": "Ты вежливый бот."}, {"role": "user", "content": f"Вопрос {i}?"},
                {"role": "assistant", "content": f"Ответ {i}."}, {"role": "user", "content": "А ещё?"},
                {"role": "assistant", "content": "Всё."}]
        return json.dumps(msgs, ensure_ascii=False) if serialized else msgs
    df = pd.DataFrame({"prompt_id": [str(i) for i in range(20)], "messages": [dialog(i) for i in range(20)]})
    rep = compat(registry, df, "text_generation")
    assert rep.ok and rep.schema_kind == "chat" and rep.input_column == "messages"
    out = frame(rep, df)
    assert (out["input"][3], out["target"][3]) == ("Ты вежливый бот.\n\nВопрос 3?", "Ответ 3.")


def test_sharegpt_roles_and_dialog_without_answer_dropped(registry):
    conv = [[{"from": "human", "value": f"Привет {i}"}, {"from": "gpt", "value": f"Здравствуйте {i}"}] for i in range(10)]
    conv[0] = [{"from": "human", "value": "только вопрос"}]
    df = pd.DataFrame({"conversations": conv})
    rep = compat(registry, df, "text_generation")
    out = frame(rep, df)
    assert rep.ok and len(out) == 9 and out["target"][0] == "Здравствуйте 1"


@pytest.mark.parametrize("description,expected", [
    ("Перевод текстов с английского на русский", ("translation.en", "translation.ru")),
    ("Перевод с русского на английский язык", ("translation.ru", "translation.en")),
])
def test_translation_dict_and_direction(registry, description, expected):
    df = pd.DataFrame({"id": [str(i) for i in range(20)],
                       "translation": [{"en": f"The cat number {i} sleeps", "ru": f"Кошка номер {i} спит"} for i in range(20)]})
    rep = compat(registry, df, "text_generation", description=description)
    assert rep.ok and (rep.input_column, rep.target_column) == expected


def test_second_reference_column_is_not_input(registry):
    """wmt24pp: source, target, original_target — во входе источник, а не второй вариант ответа."""
    df = pd.DataFrame({"lp": ["en-ru_RU"] * 20, "source": [f"English sentence number {i}" for i in range(20)],
                       "target": [f"Русское предложение {i}" for i in range(20)],
                       "original_target": [f"Русское предложение номер {i}" for i in range(20)]})
    rep = compat(registry, df, "text_generation")
    assert (rep.input_column, rep.extra_input_columns, rep.target_column) == ("source", [], "target")


def test_squad_answers_dict_as_generation_target(registry):
    df = pd.DataFrame({"title": ["Т"] * 10, "context": [f"Длинный контекст про событие {i} в истории." for i in range(10)],
                       "question": [f"Что было {i}?" for i in range(10)],
                       "answers": [{"text": np.array([f"событие {i}"]), "answer_start": np.array([25])} for i in range(10)]})
    rep = compat(registry, df, "text_generation")
    assert rep.ok and rep.target_column == "answers"
    assert [rep.input_column, *rep.extra_input_columns] == ["context", "question"]
    assert frame(rep, df)["target"][4] == "событие 4"


def test_json_answer_stays_text(registry):
    df = pd.DataFrame({"text": [f"Иван Петров, телефон 8-900-{i:03d}" for i in range(20)],
                       "output": [json.dumps({"name": "Иван Петров", "phone": f"8-900-{i:03d}"}, ensure_ascii=False)
                                  for i in range(20)]})
    rep = compat(registry, df, "text_generation")
    assert rep.ok and json.loads(frame(rep, df)["target"][5]) == {"name": "Иван Петров", "phone": "8-900-005"}


def test_summarization_unnamed_columns_short_one_is_target(registry):
    df = pd.DataFrame({"a": [f"Короткий итог {i}" for i in range(10)],
                       "b": [f"Очень длинная статья номер {i}, " * 20 for i in range(10)]})
    rep = compat(registry, df, "summarization")
    assert (rep.input_column, rep.target_column) == ("b", "a")


def test_russian_column_names(registry):
    gen = pd.DataFrame({"Вопрос клиента": [f"Где мой заказ номер {i}?" for i in range(20)],
                        "Ответ оператора": [f"Заказ {i} в пути" for i in range(20)]})
    rep = compat(registry, gen, "text_generation")
    assert (rep.input_column, rep.target_column) == ("Вопрос клиента", "Ответ оператора")
    cls = pd.DataFrame({"Текст отзыва": [f"Отзыв о товаре номер {i}, всё понравилось" for i in range(30)],
                        "Оценка": ["хорошо", "плохо", "нейтрально"] * 10, "Дата": ["2024-01-01"] * 30})
    rep = compat(registry, cls, "sentiment_analysis", ["позитив", "негатив", "нейтрально"])
    assert (rep.input_column, rep.target_column) == ("Текст отзыва", "Оценка")


# ---------------------------------------------------------------- варианты ответа

def test_multiple_choice_option_columns_letter_key(registry):
    """m_arc: вопрос в instruction, option_a…option_e (option_e пуст у вопросов на 4), ответ — буква."""
    df = pd.DataFrame({"id": [f"ARC/{i}" for i in range(12)], "answer": ["A", "B", "C", "D"] * 3,
                       "instruction": [f"Какая система органов номер {i}?" for i in range(12)],
                       "option_a": ["дыхательная"] * 12, "option_b": ["нервная"] * 12, "option_c": ["эндокринная"] * 12,
                       "option_d": ["пищеварительная"] * 12, "option_e": [None] * 12})
    rep = compat(registry, df, "text_generation")
    assert rep.ok and rep.target_column == "answer" and rep.options_columns[:4] == ["option_a", "option_b", "option_c", "option_d"]
    out = frame(rep, df)
    assert out["target"].tolist()[:4] == ["дыхательная", "нервная", "эндокринная", "пищеварительная"]
    assert out["input"][0].endswith("Варианты ответа:\n- дыхательная\n- нервная\n- эндокринная\n- пищеварительная")
    # для экстрактивного QA без контекста датасет не подходит, и LLM исправлять нечего
    rep = compat(registry, df, "qa")
    assert not rep.ok and not rep.needs_mapping


def test_multiple_choice_numbered_columns_with_context(registry):
    """belebele: flores_passage, question, mc_answer1…4, correct_answer_num с единицы."""
    df = pd.DataFrame({"link": [f"https://x/{i}" for i in range(8)], "question_number": [1, 2] * 4,
                       "flores_passage": [f"Длинный отрывок текста номер {i} о рыцарях и тамплиерах." for i in range(8)],
                       "question": [f"Вопрос {i}?" for i in range(8)],
                       "mc_answer1": ["один"] * 8, "mc_answer2": ["два"] * 8, "mc_answer3": ["три"] * 8,
                       "mc_answer4": ["четыре"] * 8, "correct_answer_num": ["1", "2", "3", "4"] * 2})
    rep = compat(registry, df, "qa")
    assert rep.ok and (rep.input_column, rep.question_column, rep.target_column) == (
        "flores_passage", "question", "correct_answer_num")
    out = frame(rep, df)
    assert out["target"].tolist()[:4] == ["один", "два", "три", "четыре"]
    assert out["input"][0].startswith("Контекст: Длинный отрывок") and "Варианты ответа:" in out["input"][0]


def test_multiple_choice_dict_with_labels_and_zero_based(registry):
    arc = pd.DataFrame({"question": [f"Вопрос про физику {i}?" for i in range(6)],
                        "choices": [{"text": ["лёд", "пар", "вода"], "label": ["1", "2", "3"]}] * 6,
                        "answerKey": ["3", "1", "2"] * 2})
    rep = compat(registry, arc, "text_generation")
    assert rep.options_columns == ["choices"] and frame(rep, arc)["target"].tolist()[:3] == ["вода", "лёд", "пар"]
    copa = pd.DataFrame({"premise": [f"Мужчина разбил окно {i}." for i in range(6)],
                         "choice1": ["Его оштрафовали."] * 6, "choice2": ["Он полетел."] * 6, "label": [0, 1] * 3})
    rep = compat(registry, copa, "text_generation")
    assert rep.target_column == "label" and frame(rep, copa)["target"].tolist()[:2] == ["Его оштрафовали.", "Он полетел."]


# ---------------------------------------------------------------- классификация

def test_label_text_preferred_over_numeric_code(registry):
    df = pd.DataFrame({"id": [str(i) for i in range(30)], "text": [f"Рецензия на фильм {i}, длинная" for i in range(30)],
                       "label": [0, 1, 2] * 10, "label_text": ["Bad", "Neutral", "Good"] * 10})
    assert compat(registry, df, "sentiment_analysis", ["позитив", "негатив"]).target_column == "label_text"


def test_spec_labels_pick_the_label_column(registry):
    df = pd.DataFrame({"text": [f"Новость номер {i} о событиях" for i in range(30)],
                       "label": [0, 1, 2] * 10, "rubric": ["спорт", "политика", "наука"] * 10})
    assert compat(registry, df, "text_classification", ["спорт", "политика", "наука"]).target_column == "rubric"


def test_multilabel_rows_dropped_single_kept(registry):
    df = pd.DataFrame({"text": [f"Комментарий номер {i}" for i in range(30)],
                       "labels": [["joy"], ["anger"], ["joy", "love"]] * 10, "id": [f"e{i}" for i in range(30)]})
    rep = compat(registry, df, "text_classification", ["joy", "anger"])
    out = frame(rep, df, "label")
    assert rep.target_column == "labels" and len(out) == 20 and set(out["target"]) == {"joy", "anger"}


def test_class_label_minus_one_is_missing_not_a_class():
    class Feat:
        names = ["negative", "positive"]
    df = _decode_class_labels(pd.DataFrame({"label": [0, 1, -1]}), {"label": Feat()})
    assert df["label"].tolist()[:2] == ["negative", "positive"] and pd.isna(df["label"][2])


def test_unlabeled_rows_dropped_before_split(registry):
    df = pd.DataFrame({"sentence": [f"Предложение номер {i}" for i in range(30)],
                       "label": ["negative", "positive", None] * 10})
    rep = compat(registry, df, "sentiment_analysis", ["positive", "negative"])
    assert rep.ok and rep.label_values == ["negative", "positive"]
    out, dropped = drop_unusable(to_task_frame(df, schema_op(rep), "label"))
    assert dropped == 10 and set(out["target"]) == {"negative", "positive"}


def test_nli_pair_with_translated_originals(registry):
    df = pd.DataFrame({"premise_original": [f"The man walks his dog number {i} in the park" for i in range(30)],
                       "hypothesis_original": [f"A man is outside with animal {i}" for i in range(30)],
                       "label": [0, 1, 2] * 10,
                       "premise": [f"Мужчина выгуливает собаку номер {i} в парке" for i in range(30)],
                       "hypothesis": [f"Мужчина на улице с животным {i}" for i in range(30)]})
    rep = compat(registry, df, "text_classification", ["entailment", "neutral", "contradiction"])
    assert [rep.input_column, *rep.extra_input_columns] == ["premise", "hypothesis"]
    assert frame(rep, df, "label")["input"][0] == "Посылка: Мужчина выгуливает собаку номер 0 в парке\nГипотеза: Мужчина на улице с животным 0"


def test_sentence_pair_numbered(registry):
    df = pd.DataFrame({"id_1": range(30), "id_2": range(30), "text_1": [f"Госдума приняла закон {i}" for i in range(30)],
                       "text_2": [f"Дума ввела запрет {i}" for i in range(30)], "class": [-1, 0, 1] * 10})
    rep = compat(registry, df, "text_classification", ["paraphrase", "not"])
    assert ([rep.input_column, *rep.extra_input_columns], rep.target_column) == (["text_1", "text_2"], "class")
    assert frame(rep, df, "label")["input"][0] == "Текст 1: Госдума приняла закон 0\nТекст 2: Дума ввела запрет 0"


def test_language_column_keeps_only_task_language(registry):
    df = pd.DataFrame({"text": [f"Текст отзыва номер {i}" if i % 2 else f"Review text number {i}" for i in range(40)],
                       "lang": ["en", "ru"] * 20, "label": ["pos", "neg", "pos", "pos"] * 10})
    rep = compat(registry, df, "sentiment_analysis", ["pos", "neg"])
    assert rep.language_filter == {"lang": "ru"}
    assert len(frame(rep, df, "label")) == 20 and all("Текст" in t for t in frame(rep, df, "label")["input"])
    other = df.assign(lang="de")
    rep = compat(registry, other, "sentiment_analysis", ["pos", "neg"])
    assert not rep.ok and any("нет строк на языке ru" in i for i in rep.issues)


# ---------------------------------------------------------------- NER

def test_ner_tag_schemes():
    toks = ["Иван", "Петров", "живёт", "в", "Москве"]
    assert bio_to_entities(toks, ["B_PER", "I_PER", "O", "O", "U-LOC"]) == [("PER", "Иван Петров"), ("LOC", "Москве")]
    assert bio_to_entities(toks, ["B-PER", "L-PER", "O", "O", "U-LOC"]) == [("PER", "Иван Петров"), ("LOC", "Москве")]
    assert bio_to_entities(toks, ["PER", "PER", "O", "O", "LOC"]) == [("PER", "Иван Петров"), ("LOC", "Москве")]


def test_ner_spans_with_offsets(registry):
    text = "Здравствуйте, Дмитрий Волков, email d.volkov@gmail.com"
    df = pd.DataFrame({"id": ["a", "b", "c"], "domain": ["HR"] * 3, "text": [text, "просто текст без данных", text],
                       "entities": [[{"start": 14, "end": 28, "text": "Дмитрий Волков", "type": "NAME"},
                                     {"start": 36, "end": 54, "text": "d.volkov@gmail.com", "type": "EMAIL"}], [],
                                    [{"start": 14, "end": 28, "type": "NAME"}]]})
    rep = compat(registry, df, "ner")
    assert rep.ok and rep.schema_kind == "ner_spans" and rep.label_values == ["EMAIL", "NAME"]
    out = frame(rep, df)
    assert out["target"].tolist() == ["NAME: Дмитрий Волков; EMAIL: d.volkov@gmail.com", "нет", "NAME: Дмитрий Волков"]


def test_ner_subword_tokens_ignored_for_char_spans(registry):
    df = pd.DataFrame({"source_text": ["Меня зовут Анна"] * 4,
                       "privacy_mask": [[{"label": "GIVENNAME", "start": 11, "end": 15, "value": "Анна"}]] * 4,
                       "mbert_tokens": [["Мен", "##я", "зовут", "Анна"]] * 4,
                       "mbert_token_classes": [["O", "O", "O", "B-GIVENNAME"]] * 4})
    rep = compat(registry, df, "ner")
    assert rep.schema_kind == "ner_spans" and (rep.input_column, rep.target_column) == ("source_text", "privacy_mask")


def test_ner_numeric_tags_without_names_explained(registry):
    df = pd.DataFrame({"tokens": [["Иван", "в", "Москве"]] * 5, "ner_tags": [[1, 0, 5]] * 5, "lang": ["ru"] * 5})
    rep = compat(registry, df, "ner")
    assert not rep.ok and "теги сущностей — числа без названий классов" in rep.issues


# ---------------------------------------------------------------- конфигурации и сплиты

def test_language_config_and_splits():
    assert pick_config(["ace_Arab", "acm_Arab", "rus_Cyrl", "eng_Latn"], "ru") == "rus_Cyrl"
    assert pick_config(["ar-ru", "de-ru", "en-ru", "en-fr"], "ru") == "en-ru"
    assert pick_config(["de-ru", "hu-ru", "en-ru"], "ru") == "en-ru"
    assert pick_config(["xquad.ar", "xquad.ru"], "ru") == "xquad.ru"
    assert pick_config(["russian", "english"], "ru") == "russian"
    assert pick_config(["default", "en"], "ru") == "default"
    assert choose_splits(["train_de", "train_ru", "val_ru", "test_ru"], "ru") == ["train_ru", "val_ru", "test_ru"]
    assert choose_splits(["ar_mnli", "ru_mnli", "ru_fever"], "ru") == ["ru_mnli", "ru_fever"]
    assert choose_splits(["train", "test"], "ru") is None


# ---------------------------------------------------------------- правка сопоставления от LLM

def _unnamed_qa():
    return pd.DataFrame({"context": [f"Пушкин родился в 1799 году в Москве, текст {i}." for i in range(10)],
                         "question": [f"Где родился Пушкин? ({i})" for i in range(10)],
                         "resp_text": [f"в Москве ({i})" for i in range(10)]})


def test_unrecognized_answer_column_is_left_to_llm(registry):
    rep = compat(registry, _unnamed_qa(), "qa")
    assert not rep.ok and rep.needs_mapping
    prompt = build_prompt(spec(registry, "qa"), _cand(rep), _unnamed_qa(), 3)
    assert "не определено автоматически" in prompt and "resp_text (" in prompt


def _cand(rep):
    from factory.core.contracts import DatasetCandidate
    return DatasetCandidate(hf_id="x/y", compat=rep, license="mit")


def test_llm_mapping_fix_is_validated(registry):
    df, sp, task = _unnamed_qa(), spec(registry, "qa"), registry.task("qa")
    cand = _cand(compat(registry, df, "qa"))
    fixed = apply_mapping_fix(sp, task, cand, df, MappingFix(input_columns=["context"], question_column="question",
                                                               target_column="resp_text"))
    assert fixed and fixed.ok and fixed.target_column == "resp_text" and fixed.preview[0]["target"] == "в Москве (0)"
    # несуществующая колонка и ответ из входа — отклоняются
    assert apply_mapping_fix(sp, task, cand, df, MappingFix(input_columns=["context"], target_column="нет такой")) is None
    assert apply_mapping_fix(sp, task, cand, df, MappingFix(input_columns=["context"], target_column="context")) is None


def test_compat_preview_shows_task_examples(registry):
    df = pd.DataFrame({"instruction": RU * 2, "output": [f"ответ {i}" for i in range(12)]})
    rep = compat(registry, df, "text_generation")
    assert rep.preview[0] == {"input": RU[0], "target": "ответ 0"} and len(rep.preview) == 5


def test_assess_applies_fix_and_rescores(registry):
    """LLM исправила сопоставление — оценки берутся из повторной оценки по исправленным примерам."""
    import asyncio

    from pydantic_ai import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from factory.core import llm
    from factory.search.llm_eval import CRITERIA, assess

    prompts = []

    def fn(messages, info: AgentInfo):
        prompt = next(p.content for m in messages for p in getattr(m, "parts", []) if getattr(p, "part_kind", "") == "user-prompt")
        prompts.append(prompt)
        first = len(prompts) == 1
        args = {k: {"score": 1 if first else 5, "reason": "r"} for k in CRITERIA} | {"why": "w"}
        if first:
            args |= {"mapping_ok": False, "mapping_fix": {"input_columns": ["prompt"], "target_column": "text"}}
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=args)])

    qs = [f"Вопрос номер {i} про физику?" for i in range(12)]
    df = pd.DataFrame({"text": [f"Ответ {i}: так устроена природа" for i in range(12)], "prompt": qs})
    sp = spec(registry, "text_generation")
    cand = _cand(compat(registry, df, "text_generation"))
    assert (cand.compat.input_column, cand.compat.target_column) == ("text", "prompt")  # эвристика ошиблась
    with llm.override_model(FunctionModel(fn)):
        out = asyncio.run(assess(sp, registry.task("text_generation"), cand, df, {k: 1.0 for k in CRITERIA}))
    assert (out.compat.input_column, out.compat.target_column) == ("prompt", "text") and out.score == 100.0
    assert len(prompts) == 2 and '"ответ": "Ответ 0' in prompts[1]


def test_file_sorted_by_label(registry):
    """Файл пользователя отсортирован по метке: первые сотни строк — один класс. Колонка меток всё равно найдена,
    а в примерах для LLM — разные метки."""
    df = pd.DataFrame({"text": [f"Отзыв номер {i}, хороший товар" for i in range(300)] + [f"Отзыв номер {i}, плохой товар" for i in range(300)],
                       "label": ["позитив"] * 300 + ["негатив"] * 300})
    rep = compat(registry, df, "sentiment_analysis", ["позитив", "негатив"])
    assert rep.ok and rep.target_column == "label" and {e["target"] for e in rep.preview} == {"позитив", "негатив"}
