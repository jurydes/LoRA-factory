"""Сопоставление колонок датасета с форматом задачи: что модель видит на входе и что учится отвечать.

Колонки распознаются по содержимому (текст, категория, список токенов, диалог, сущности с позициями, варианты
ответа, сериализованные в строку структуры) и по имени (словари ролей, в том числе русские). Решение
детерминированное; LLM-оценка датасета видит получившиеся примеры и может предложить другое сопоставление —
его проверяет код пробным приведением (search/llm_eval).

Формат задачи — колонки: _id (стабильный хэш содержимого), input, target, synthetic; для QA ещё alt (все эталоны).
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from factory.core.contracts import MapSchemaOp, TaskFamily
from factory.prep import lang

MAX_CLASSES = 200
PROBE_ROWS = 200  # по скольким строкам определяется вид колонки
NO_ENTITIES = "нет"

# ---------------------------------------------------------------- языки

LANG_ALIASES = {
    "ru": ["ru", "rus", "russian", "русский"], "en": ["en", "eng", "english", "английский"],
    "uk": ["uk", "ukr", "ukrainian", "украинский"], "be": ["be", "bel", "belarusian"], "kk": ["kk", "kaz", "kazakh"],
    "tt": ["tt", "tat", "tatar"], "de": ["de", "deu", "ger", "german"], "fr": ["fr", "fra", "fre", "french"],
    "es": ["es", "spa", "spanish"], "it": ["it", "ita", "italian"], "pt": ["pt", "por", "portuguese"],
    "pl": ["pl", "pol", "polish"], "cs": ["cs", "ces", "cze", "czech"], "nl": ["nl", "nld", "dut", "dutch"],
    "tr": ["tr", "tur", "turkish"], "ar": ["ar", "ara", "arb", "arabic"], "zh": ["zh", "zho", "chi", "cmn", "chinese"],
    "ja": ["ja", "jpn", "japanese"], "ko": ["ko", "kor", "korean"], "hi": ["hi", "hin", "hindi"],
    "vi": ["vi", "vie", "vietnamese"], "fa": ["fa", "fas", "persian"], "he": ["he", "heb", "hebrew"],
}
_ALIAS = {a: code for code, names in LANG_ALIASES.items() for a in names}
# язык, на который переводят, по постановке: «перевод с русского на английский», «translate into English»
_TARGET_LANG_STEMS = {"русск": "ru", "английск": "en", "украинск": "uk", "белорусск": "be", "казахск": "kk",
                      "татарск": "tt", "немецк": "de", "французск": "fr", "испанск": "es", "итальянск": "it",
                      "португальск": "pt", "польск": "pl", "чешск": "cs", "турецк": "tr", "арабск": "ar",
                      "китайск": "zh", "японск": "ja", "корейск": "ko", "russian": "ru", "english": "en",
                      "german": "de", "french": "fr", "spanish": "es", "ukrainian": "uk", "chinese": "zh"}


def tokens(name) -> list[str]:
    """Слова имени: answerKey → answer key, mc_answer1 → mc answer 1, translation.ru → translation ru."""
    s = re.sub(r"([a-zа-яё])([A-ZА-ЯЁ])", r"\1 \2", str(name))
    s = re.sub(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])", " ", s)
    return [t for t in re.split(r"[\W_]+", s.lower()) if t]


def lang_code(name) -> str | None:
    """ISO-код по названию языка, коду ISO 639-1/3 или метке вида rus_Cyrl, ru-RU; None — не язык или несколько."""
    codes = {_ALIAS[t] for t in tokens(name) if t in _ALIAS}
    return codes.pop() if len(codes) == 1 else None


def mentions_language(name, language: str) -> bool:
    return language in {_ALIAS[t] for t in tokens(name) if t in _ALIAS}


def target_language(description: str) -> str | None:
    m = re.search(r"\bна\s+([а-яё]+)|\b(?:into|to)\s+([a-z]+)", str(description).lower())
    word = m and (m.group(1) or m.group(2))
    return next((code for stem, code in _TARGET_LANG_STEMS.items() if word and word.startswith(stem)), None)


# ---------------------------------------------------------------- словари ролей по именам колонок
# Слово целиком (и с окончанием -s/-es) или основа со «*». Порядок — приоритет.

META = ["id", "ids", "idx", "index", "uid", "uuid", "guid", "hash", "md5", "sha*", "url*", "link*", "href", "uri",
        "date*", "time*", "year", "month", "split", "lang", "langs", "language*", "locale", "dialect", "license",
        "num", "number", "cluster*", "embedding*", "emb", "vector*", "prob*", "length", "len", "count", "offset*",
        "start", "end", "pos", "upos", "xpos", "chunk*", "lemma*", "deprel", "дата", "ссылк*"]
INPUT = ["input*", "text", "sentence*", "content", "document", "article", "body", "source", "src", "comment*",
         "review*", "query", "prompt", "message", "utterance", "post", "tweet", "question", "passage", "original",
         "complex", "toxic*", "premise", "x", "текст*", "отзыв*", "коммент*", "сообщен*", "стать*", "документ*",
         "запрос*", "вопрос*", "исходн*", "вход*", "обращен*"]
INSTRUCTION = ["system*", "instruction*", "инструкц*", "систем*"]
TITLE = ["title*", "headline*", "subject", "заголов*"]
CONTEXT = ["context*", "passage*", "paragraph*", "support*", "evidence", "background", "story", "premise*",
           "контекст*", "отрыв*", "фрагмент*"]
QUESTION = ["question*", "query", "queries", "hypothes*", "claim", "q", "вопрос*", "запрос*", "гипотез*"]
QA_CONTEXT = CONTEXT + ["document*", "article*", "text*", "body", "content", "документ*", "текст*", "стать*"]
QA_QUESTION = QUESTION + ["instruction*", "prompt", "инструкц*"]
ANSWER = ["answer*", "ответ*", "gold*", "correct*", "target*", "label*", "output*", "response*", "solution*"]
TARGET_CLS = ["label*", "class*", "category*", "categories", "sentiment*", "target*", "topic*", "intent*",
              "emotion*", "polarity", "rating*", "stars", "toxic*", "tag", "y", "output*", "answer*", "метк*",
              "класс*", "категор*", "тональн*", "тем*", "рейтинг*", "оценк*", "сентимент*"]
TARGET_GEN = ["target*", "output*", "response*", "answer*", "completion*", "reply", "neutral*", "simpl*",
              "paraphras*", "detox*", "rewrit*", "correct*", "chosen", "tgt", "reference*", "gold*", "result*",
              "translation*", "summar*", "y", "ответ*", "перевод*", "результат*", "исправлен*", "упрощ*",
              "нейтральн*", "перефраз*", "выход*", "цел*"]
TARGET_SUM = ["summar*", "abstract*", "highlight*", "headline*", "tldr", "gist", "annotation*", "резюме",
              "аннотац*", "реферат*", "кратк*", "выжимк*", "title*", "заголов*", "target*", "output*",
              "response*", "answer*"]
TOKENS = ["tokens", "token", "words", "word", "токен*", "слов*"]
TAGS = ["ner*", "tag*", "label*", "bio", "iob*", "entit*", "сущност*", "метк*"]
SPANS = ["entit*", "span*", "mask*", "ner*", "annotation*", "label*", "сущност*"]
OPTIONS = ["choice*", "option*", "variant*", "ending*", "candidate*", "alternative*", "вариант*"]
OPTION_BASES = OPTIONS + ["answer*", "sol*", "ответ*"]  # option_a…e, mc_answer1…4, sol1/sol2
CHAT = ["messages", "conversation*", "dialog*", "chat*", "turns", "диалог*"]
LANG_COLUMN = ["lang", "langs", "language*", "locale", "dialect", "язык*"]


def match(toks: list[str], vocab: list[str]) -> int | None:
    """Позиция первого слова словаря, которое есть в имени; None — имя не подходит."""
    for i, w in enumerate(vocab):
        stem = w.rstrip("*")
        for t in toks:
            if t.startswith(stem) if w.endswith("*") else t in (w, w + "s", w + "es"):
                return i
    return None


def _extra(toks: list[str], vocab: list[str]) -> int:
    """Сколько слов имени не из словаря: «target» точнее «original_target»."""
    return sum(match([t], vocab) is None for t in toks)


# ---------------------------------------------------------------- значения

def _is_seq(v) -> bool:
    return isinstance(v, (list, tuple, np.ndarray))


def _missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and not v.strip()


def _as_text(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def parsed(v):
    """Словарь или список, сохранённый строкой (JSON или repr Python), → объект; иначе значение как есть."""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if len(s) < 2 or s[0] not in "{[" or s[-1] not in "}]":
        return v
    try:
        return json.loads(s)
    except ValueError:
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
            return v


def _vtype(v) -> str:
    if isinstance(v, str):
        return "str"
    if isinstance(v, (bool, np.bool_)):
        return "bool"
    if isinstance(v, (int, np.integer)):
        return "int"
    if isinstance(v, (float, np.floating)):
        return "float"
    if isinstance(v, (datetime, date, pd.Timestamp, np.datetime64)):
        return "date"
    if isinstance(v, dict):
        return "dict"
    if _is_seq(v):
        return "list"
    return "other"


ROLE_KEYS = ("role", "from", "speaker", "author", "sender")
CONTENT_KEYS = ("content", "value", "text", "message")
SPAN_TYPE_KEYS = ("type", "label", "entity_group", "entity", "tag", "category", "entity_type")
SPAN_TEXT_KEYS = ("text", "value", "word", "span", "mention")


def _list_kind(values: list) -> str:
    elems = [x for v in values for x in list(v)][:2000]
    if not elems:
        return "empty_list"
    kinds = Counter(_vtype(x) for x in elems)
    top, n = kinds.most_common(1)[0]
    if n < 0.9 * len(elems):
        return "list"
    if top == "dict":
        keys = Counter(k for x in elems for k in x)
        has = lambda names: any(keys[k] >= 0.8 * len(elems) for k in names)
        if has(ROLE_KEYS) and has(CONTENT_KEYS):
            return "messages"
        if has(SPAN_TYPE_KEYS) and (has(("start", "begin", "start_offset")) or has(SPAN_TEXT_KEYS)):
            return "spans"
        return "dict_list"
    return {"str": "str_list", "int": "int_list", "bool": "int_list", "float": "float_list"}.get(top, "list")


@dataclass
class Col:
    """Колонка датасета: имя, вид содержимого и до PROBE_ROWS непустых значений (строки-контейнеры разобраны)."""
    name: str
    kind: str  # text | category | constant | url | number | date | dict | str_list | int_list | float_list |
    #            messages | spans | dict_list | empty_list | empty | list | mixed | other
    values: list
    serialized: bool = False  # значения — структуры, сохранённые строкой
    fill: float = 1.0  # доля непустых значений
    _lang: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.toks = tokens(self.name)
        self.meta = match(self.toks, META) is not None
        self.strings = [_as_text(v).strip() for v in self.values] if not self.serialized else []
        self.mean_len = float(np.mean([len(s) for s in self.strings])) if self.strings else 0.0

    @property
    def is_text(self) -> bool:
        """Свободный текст (или структура строкой: для генерации это может быть и ответ в JSON)."""
        return (self.kind == "text" or self.serialized) and not self.meta

    def lang_ok(self, language: str | None) -> bool:
        """Тексты колонки — на языке задачи (большинство); короткие и нетекстовые не проверяются."""
        if not language or not self.strings or self.mean_len < 8:
            return True
        if language not in self._lang:
            sample = [s for s in self.strings if len(s) >= 3][:30]
            self._lang[language] = not sample or lang.language_share(sample).get(language, 0.0) >= 0.5
        return self._lang[language]


def _spread(items: list, k: int) -> list:
    """k элементов равномерно по всему списку: файлы бывают отсортированы по метке — первые строки одного класса."""
    if len(items) <= k:
        return items
    return [items[int(i * len(items) / k)] for i in range(k)]


def _column(name: str, series: pd.Series) -> Col:
    raw = _spread(list(series), PROBE_ROWS * 3)
    present = [v for v in raw if not _missing(v)]
    fill = len(present) / max(len(raw), 1)
    values = _spread(present, PROBE_ROWS)
    if not values:
        return Col(name, "empty", [], fill=fill)
    serialized = False
    if all(isinstance(v, str) for v in values):
        objs = [parsed(v) for v in values]
        if sum(not isinstance(o, str) for o in objs) >= 0.9 * len(objs):
            values, serialized = [o for o in objs if not isinstance(o, str)], True
    kinds = Counter(_vtype(v) for v in values)
    top, n = kinds.most_common(1)[0]
    if n < 0.9 * len(values):
        return Col(name, "mixed", values, serialized, fill)
    if top == "str":
        s = [v.strip() for v in values]
        if sum(bool(re.match(r"(https?://|www\.)\S+$", x)) for x in s) >= 0.8 * len(s):
            return Col(name, "url", values, fill=fill)
        u = len(set(s))
        if u == 1 and len(s) >= 3:
            return Col(name, "constant", values, fill=fill)
        mean_len = sum(map(len, s)) / len(s)
        words = sum(len(x.split()) for x in s) / len(s)
        # категория: значения повторяются и они короткие (метка — 1–4 слова) или повторяются заметно
        if u <= MAX_CLASSES and u < len(s) and u <= max(30, len(s) // 2) and mean_len <= 60 and (
                words <= 4 or u <= len(s) / 3):
            return Col(name, "category", values, fill=fill)
        return Col(name, "text", values, fill=fill)
    if top in ("int", "bool", "float"):
        nums = [v for v in values if _vtype(v) == top]
        u = len(set(nums))
        integral = top != "float" or all(float(v).is_integer() for v in nums)
        if u == 1 and len(nums) >= 3:
            return Col(name, "constant", values, fill=fill)
        if integral and u <= MAX_CLASSES and u <= max(30, len(nums) // 2):
            return Col(name, "category", values, fill=fill)
        return Col(name, "number", values, fill=fill)
    if top == "list":
        return Col(name, _list_kind(values), values, serialized, fill)
    return Col(name, {"date": "date", "dict": "dict"}.get(top, "other"), values, serialized, fill)


def profile_columns(df: pd.DataFrame) -> list[Col]:
    return [_column(str(c), df[c]) for c in df.columns if not str(c).startswith("_")]


def flatten_dicts(df: pd.DataFrame) -> pd.DataFrame:
    """Колонки-словари → ещё колонки «родитель.ключ» (перевод {en, ru}, варианты {text, label}, ответы SQuAD).
    Исходная колонка остаётся. Повторный вызов ничего не меняет."""
    out = df
    for c in list(df.columns):
        if str(c).startswith("_") or df[c].dtype != object:
            continue
        vals = [v for v in df[c].head(PROBE_ROWS) if not _missing(v)]
        if not vals or not all(isinstance(v, dict) for v in vals):
            continue
        keys = Counter(k for v in vals for k in v)
        for k, n in keys.most_common(20):
            name = f"{c}.{k}"
            if n < 0.5 * len(vals) or name in out.columns:
                continue
            if out is df:
                out = df.copy(deep=False)
            out[name] = [v.get(k) if isinstance(v, dict) else None for v in df[c]]
    return out


# ---------------------------------------------------------------- метки, ответы, варианты

def normalize_label(s: str) -> str:
    return re.sub(r"[\s\W_]+", " ", str(s).lower()).strip()


def label_value(v, single: bool = True) -> str:
    """Метка класса строкой: 1.0 → «1»; список из одной метки → она; несколько меток (multi-label) → «»,
    если нужна одна (single), иначе первая."""
    v = parsed(v) if isinstance(v, str) else v
    if _is_seq(v):
        items = [x for x in (label_value(x) for x in list(v)) if x]
        return items[0] if items and (len(items) == 1 or not single) else ""
    if isinstance(v, (float, np.floating)):
        return "" if math.isnan(v) else (str(int(v)) if float(v).is_integer() else str(v))
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    return _as_text(v).strip()


def answers(v) -> list[str]:
    """Ответы QA: SQuAD {"text": [...]}, DROP {"number", "date", "spans"} (по разметчикам), список или строка —
    в том числе словарь или список, сохранённые строкой (repr/JSON): иначе целью стала бы запись всего словаря."""
    v = parsed(v)
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    if isinstance(v, dict):
        if "text" in v or "answer" in v:
            return answers(v.get("text", v.get("answer")))
        if {"day", "month", "year"} <= set(v):  # дата DROP
            d = " ".join(str(v[k]).strip() for k in ("day", "month", "year") if str(v[k]).strip())
            return [d] if d else []
        if {"number", "spans", "date"} & set(v):  # DROP: фрагменты, число или дата
            out = answers(v.get("spans")) + answers(v.get("number")) + answers(v.get("date"))
            return list(dict.fromkeys(out))
        return []
    if _is_seq(v):
        return list(dict.fromkeys(a for x in list(v) for a in answers(x)))
    t = label_value(v)
    return [t] if t else []


def _suffix(toks: list[str]) -> tuple[list[str], str | None]:
    """option_a → ([option], a); mc_answer3 → ([mc, answer], 3); A → ([], a)."""
    if toks and (re.fullmatch(r"[a-z]", toks[-1]) or re.fullmatch(r"\d{1,2}", toks[-1])):
        return toks[:-1], toks[-1]
    return toks, None


def _run(suffixes: list[str]) -> bool:
    """Суффиксы подряд от a / 0 / 1: a b c d или 1 2 3 4 — варианты; x y — нет."""
    if all(s.isdigit() for s in suffixes):
        nums = sorted(int(s) for s in suffixes)
        return nums[0] in (0, 1) and nums == list(range(nums[0], nums[0] + len(nums)))
    return sorted(suffixes) == [chr(97 + i) for i in range(len(suffixes))]


def _choice_texts(v) -> tuple[list[str], list[str] | None]:
    """Варианты из колонки-списка или словаря {text: [...], label: [...]}."""
    v = parsed(v)
    if isinstance(v, dict):
        texts = next((v[k] for k in ("text", "texts", "choices", "options") if k in v), None)
        labels = next((v[k] for k in ("label", "labels", "keys") if k in v), None)
        texts = [_as_text(x).strip() for x in list(texts)] if _is_seq(texts) else []
        labels = [_as_text(x).strip() for x in list(labels)] if _is_seq(labels) and len(labels) == len(texts) else None
        return texts, labels
    if _is_seq(v):
        return [_as_text(x).strip() for x in list(v)], None
    return [], None


def options_per_row(df: pd.DataFrame, columns: list[str]) -> list[tuple[list[str], list[str] | None]]:
    """[(тексты вариантов, их ключи или None)] по строкам; пустые варианты (option_e у вопросов на 4) пропускаются."""
    if len(columns) == 1:
        rows = [_choice_texts(v) for v in df[columns[0]]]
    else:
        keys = [_suffix(tokens(c))[1] or c for c in columns]
        rows = [([_as_text(v).strip() for v in vals], keys) for vals in zip(*(df[c] for c in columns))]
    out = []
    for texts, labels in rows:
        keep = [i for i, t in enumerate(texts) if t and t.lower() not in ("none", "nan")]
        out.append(([texts[i] for i in keep], [labels[i] for i in keep] if labels else None))
    return out


def _norm_key(k) -> str:
    return str(k).strip("().: ").lower()


def resolve_choices(keys, options: list[tuple[list[str], list[str] | None]]) -> list[str]:
    """Ключи правильных вариантов по строкам → тексты. Как читать ключи, решается по всей колонке: метки
    вариантов — только если все ключи среди них (COPA: choice1/choice2, а label считает с нуля); цифры —
    с нуля, если среди ключей есть 0, иначе с единицы."""
    keys = [label_value(k) for k in keys]
    labels = {_norm_key(lab) for _, labs in options for lab in labs or []}
    use_labels = bool(labels) and all(_norm_key(k) in labels for k in keys if k)
    base = 0 if any(k == "0" for k in keys) else 1
    return [resolve_choice(k, texts, labs if use_labels else None, base) for k, (texts, labs) in zip(keys, options)]


def resolve_choice(key, texts: list[str], labels: list[str] | None, base: int = 1) -> str:
    """Ключ правильного варианта (A, 2, «(c)» или сам текст) → текст варианта; «» — не удалось."""
    k = label_value(key)
    if not texts or not k:
        return ""
    if k in texts:
        return k
    kn = _norm_key(k)
    if labels:
        for t, lab in zip(texts, labels):
            if _norm_key(lab) == kn:
                return t
    if re.fullmatch(r"[a-z]", kn):
        i = ord(kn) - 97
    elif kn.isdigit():
        i = int(kn) - base
    else:
        return ""
    return texts[i] if 0 <= i < len(texts) else ""


def options_block(texts: list[str]) -> str:
    return "Варианты ответа:\n" + "\n".join(f"- {t}" for t in texts)


# ---------------------------------------------------------------- NER

def detokenize(toks) -> str:
    text = " ".join(str(t) for t in toks)
    return re.sub(r"\s+([.,!?:;»)\]])", r"\1", re.sub(r"([«(\[])\s+", r"\1", text))


def _split_tag(tag: str) -> tuple[str | None, str | None]:
    tag = str(tag).strip()
    if tag in ("O", "0", "") or tag.lower() == "o":
        return None, None
    m = re.fullmatch(r"([BIESLU])[-_](.+)", tag)
    if m:
        return {"L": "E", "U": "S"}.get(m.group(1), m.group(1)), m.group(2)
    return "I", tag  # схема IO: тип без префикса


def bio_to_entities(toks, tags) -> list[tuple[str, str]]:
    """BIO/IOB/BIOES/BILOU-теги (через «-» или «_») и IO без префиксов → [(тип, текст)].
    I- без B- начинает новую сущность (IOB1)."""
    out, cur_type, cur = [], None, []

    def flush():
        if cur_type and cur:
            out.append((cur_type, detokenize(cur)))
    for tok, tag in zip(list(toks), list(tags)):
        prefix, typ = _split_tag(tag)
        if typ is None:
            flush()
            cur_type, cur = None, []
            continue
        if prefix in ("B", "S") or typ != cur_type:
            flush()
            cur_type, cur = typ, [tok]
        else:
            cur.append(tok)
        if prefix in ("E", "S"):
            flush()
            cur_type, cur = None, []
    flush()
    return out


def spans_to_entities(text: str, spans) -> list[tuple[str, str]]:
    """Сущности с позициями ({start, end, type/label} или {text, type}) или строки «ТИП: текст» → [(тип, текст)]
    в порядке появления в тексте."""
    spans = parsed(spans)
    if not _is_seq(spans):
        return []
    found = []
    for s in list(spans):
        if isinstance(s, str) and ":" in s:  # уже «ТИП: текст» (wikiann spans)
            typ, ent = s.split(":", 1)
            found.append((text.find(ent.strip()), typ.strip(), ent.strip()))
            continue
        if not isinstance(s, dict):
            continue
        typ = next((_as_text(s[k]).strip() for k in SPAN_TYPE_KEYS if k in s and _as_text(s[k]).strip()), "")
        start = next((s[k] for k in ("start", "begin", "start_offset") if isinstance(s.get(k), (int, np.integer))), None)
        end = next((s[k] for k in ("end", "stop", "end_offset") if isinstance(s.get(k), (int, np.integer))), None)
        ent = next((_as_text(s[k]).strip() for k in SPAN_TEXT_KEYS if k in s and _as_text(s[k]).strip()), "")
        if not ent and start is not None and end is not None:
            ent = text[int(start):int(end)].strip()
        if typ and ent:
            found.append((int(start) if start is not None else text.find(ent), typ, ent))
    found.sort(key=lambda x: (x[0] < 0, x[0]))
    return [(t, e) for _, t, e in found]


def format_entities(ents: list[tuple[str, str]]) -> str:
    return "; ".join(f"{t}: {x}" for t, x in ents) if ents else NO_ENTITIES


def _tag_like(c: Col) -> bool:
    elems = [str(x) for v in c.values for x in list(v)]
    if not elems or any(" " in x or len(x) > 40 for x in elems):
        return False
    if all(re.fullmatch(r"(?:[BIESLU][-_]\S+|O)", x) for x in elems):
        return True
    return elems.count("O") >= 0.2 * len(elems)  # IO: «O» и типы без префиксов


def _subword(c: Col) -> bool:
    elems = [str(x) for v in c.values for x in list(v)]
    return sum(x.startswith(("##", "▁", "Ġ")) for x in elems) >= 0.05 * max(len(elems), 1)


# ---------------------------------------------------------------- диалоги

USER_ROLES = {"user", "human", "prompter", "client", "customer", "usr", "пользователь", "клиент"}
ASSISTANT_ROLES = {"assistant", "gpt", "bot", "model", "chatbot", "ai", "operator", "agent", "ассистент", "бот",
                   "оператор"}
SYSTEM_ROLES = {"system", "система"}


def chat_pair(messages) -> tuple[str, str]:
    """Первый обмен репликами: (системная инструкция + реплики пользователя до первого ответа, ответ ассистента).
    Задаче нужен один ответ на один запрос — длинная история диалога во входе не помогает."""
    messages = parsed(messages)
    system, user = [], []
    for m in list(messages) if _is_seq(messages) else []:
        if not isinstance(m, dict):
            continue
        role = next((_as_text(m[k]).strip().lower() for k in ROLE_KEYS if k in m), "")
        text = next((_as_text(m[k]).strip() for k in CONTENT_KEYS if k in m), "")
        if not text:
            continue
        if role in SYSTEM_ROLES and not user:
            system.append(text)
        elif role in ASSISTANT_ROLES:
            if user:
                return "\n\n".join(system + ["\n".join(user)]), text
        else:
            user.append(text)
    return "\n\n".join(system + ["\n".join(user)]) if user else "", ""


# ---------------------------------------------------------------- выбор колонок

def _language_filter(cols: list[Col], language: str | None) -> dict[str, str]:
    """Колонки языка строки (lang, language, prompt_lang…): в датасете смешаны языки — оставляем строки нужного."""
    if not language:
        return {}
    out = {}
    for c in cols:
        if c.kind not in ("category", "constant", "text") or match(c.toks, LANG_COLUMN) is None:
            continue
        codes = [lang_code(s) for s in c.strings]
        if codes and sum(x is not None for x in codes) >= 0.8 * len(codes):
            out[c.name] = language
    return out


def filter_language(df: pd.DataFrame, language_filter: dict[str, str]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, code in language_filter.items():
        mask &= df[col].map(lambda v, code=code: lang_code(_as_text(v)) == code)
    return df[mask]


def _leaks(c: Col, target: Col | None, df: pd.DataFrame) -> bool:
    """Колонка содержит сам ответ (alpaca: text = инструкция + вход + ответ) — во вход её брать нельзя."""
    if target is None or not target.strings or target.mean_len < 8:
        return False
    pairs = [(_as_text(a), _as_text(b).strip()) for a, b in zip(df[c.name].head(PROBE_ROWS), df[target.name].head(PROBE_ROWS))]
    pairs = [(a, b) for a, b in pairs if b]
    return bool(pairs) and sum(b in a for a, b in pairs) >= 0.5 * len(pairs)


def input_role(name: str) -> tuple[int, str]:
    """Порядок колонки во входе и её подпись: инструкция → заголовок → контекст → текст → вопрос."""
    toks = tokens(name)
    _, suf = _suffix(toks)
    num = f" {suf}" if suf and suf.isdigit() else ""
    if match(toks, INSTRUCTION) is not None:
        return (0, "Системная инструкция") if any(t.startswith(("system", "систем")) for t in toks) else (1, "Инструкция")
    if match(toks, TITLE) is not None:
        return 2, "Заголовок"
    if match(toks, CONTEXT) is not None:
        return 3, ("Посылка" if match(toks, ["premise*"]) is not None else "Контекст")
    if match(toks, QUESTION) is not None:
        if match(toks, ["hypothes*", "гипотез*"]) is not None:
            return 5, "Гипотеза"
        return 5, ("Утверждение" if "claim" in toks else "Вопрос") + num
    return 4, "Текст" + num


def render_inputs(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Одна колонка — значение как есть; несколько — строки «Подпись: значение» (пустые пропускаются)."""
    if len(columns) == 1:
        return [_as_text(v).strip() for v in df[columns[0]]]
    labels = [input_role(c)[1] for c in columns]
    if len(set(labels)) < len(labels):  # одинаковые подписи (две колонки без роли) — подписываем именами
        labels = list(columns)
    return ["\n".join(f"{lab}: {_as_text(v).strip()}" for lab, v in zip(labels, vals) if _as_text(v).strip())
            for vals in zip(*(df[c] for c in columns))]


def _companions(main: Col, texts: list[Col], language: str | None, family: TaskFamily) -> list[Col]:
    """Другие колонки входа: пара с тем же именем (sentence1/sentence2), инструкция, контекст, вопрос,
    посылка/гипотеза; для классификации ещё заголовок. Колонки на другом языке (перевод того же) не берутся."""
    base, suf = _suffix(main.toks)
    out = []
    for c in texts:
        if c is main or (main.lang_ok(language) and not c.lang_ok(language)):
            continue
        b, s = _suffix(c.toks)
        paired = suf is not None and s is not None and b == base
        role = any(match(c.toks, v) is not None for v in (INSTRUCTION, CONTEXT, QUESTION))
        if paired or role or (family == "classification" and match(c.toks, TITLE) is not None):
            out.append(c)
    return out


def phrases(c: Col) -> bool:
    """Повторяющиеся фразы (в выборке есть дубли строк): по виду категория, по сути — текст. Метки
    классов — короче (1–2 слова)."""
    return (c.kind == "category" and not c.meta and bool(c.strings) and all(isinstance(v, str) for v in c.values)
            and sum(len(s.split()) for s in c.strings) / len(c.strings) >= 3)


def _wordy(c: Col) -> bool:
    """Повторяющиеся короткие тексты (шаблонные инструкции, названия) с «входным» именем — тоже вход."""
    named = any(match(c.toks, v) is not None for v in (INPUT, INSTRUCTION, CONTEXT, QUESTION))
    return (c.kind == "category" and not c.meta and all(isinstance(v, str) for v in c.values)
            and any(re.search(r"[^\W\d_]", s) for s in c.strings) and (phrases(c) or named))


VARIANT = {"original", "orig", "raw", "clean", "cleaned", "translated", "translation", "normalized", "norm",
           "processed", "lower", "lowercased"}


def _one_per_field(texts: list[Col], language: str | None) -> list[Col]:
    """premise / premise_original / premise_en — одно поле в разных вариантах: берём то, что на языке задачи."""
    groups: dict[tuple, list[Col]] = defaultdict(list)
    for c in texts:
        groups[tuple(t for t in c.toks if t not in VARIANT and t not in _ALIAS)].append(c)
    keep = {min(g, key=lambda c: (not c.lang_ok(language), len(c.toks), texts.index(c))).name for g in groups.values()}
    return [c for c in texts if c.name in keep]


def _inputs(cols: list[Col], df: pd.DataFrame, exclude: set[str], language: str | None, family: TaskFamily,
            target: Col | None = None, answer_vocab: list[str] | None = None) -> list[str]:
    """Колонки входа в порядке подачи. answer_vocab — колонки с такими именами (второй вариант ответа:
    original_target) во вход не берутся, пока есть другие."""
    texts = [c for c in cols if c.name not in exclude and (c.is_text or _wordy(c)) and c.fill >= 0.05
             and not _leaks(c, target, df)]
    if answer_vocab:
        texts = [c for c in texts if match(c.toks, answer_vocab) is None] or texts
    texts = _one_per_field(texts, language)
    if not texts:
        return []

    def key(c: Col):
        rank = match(c.toks, INPUT)
        return not c.is_text, not c.lang_ok(language), rank is None, rank or 0, _extra(c.toks, INPUT), -c.mean_len
    main = min(texts, key=key)
    chosen = [main] + _companions(main, texts, language, family)
    order = {c.name: i for i, c in enumerate(cols)}
    return [c.name for c in sorted(chosen, key=lambda c: (input_role(c.name)[0], order[c.name]))]


def _key_like(c: Col) -> bool:
    vals = [label_value(v) for v in c.values]
    return bool(vals) and len(set(vals)) <= 26 and sum(
        bool(re.fullmatch(r"\(?[A-Za-z]\)?\.?|\d{1,2}", v)) for v in vals) >= 0.9 * len(vals)


def _multiple_choice(cols: list[Col]) -> tuple[Col, list[str]] | None:
    """Ключ правильного варианта (A/B/…, 1/2/…) и колонки вариантов (список, словарь {text, label} или
    option_a…option_e, mc_answer1…4)."""
    groups: list[list[str]] = []
    for c in cols:
        if "." not in c.name and match(c.toks, OPTIONS) is not None and (
                c.kind == "str_list" or (c.kind == "dict" and any(_choice_texts(v)[0] for v in c.values))):
            groups.append([c.name])
    by_base: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for c in cols:
        base, suf = _suffix(c.toks)
        if "." not in c.name and suf is not None and c.kind in ("text", "category", "constant", "empty") and (
                not base or match(base, OPTION_BASES) is not None):
            by_base[tuple(base)].append((suf, c.name))
    for members in by_base.values():
        if len(members) >= 2 and _run([s for s, _ in members]):
            groups.append([n for _, n in sorted(members, key=lambda x: (len(x[0]), x[0]))])
    for group in groups:
        keys = [c for c in cols if c.name not in group and "." not in c.name and _key_like(c)
                and match(c.toks, ANSWER + ["key"]) is not None]
        if keys:
            return min(keys, key=lambda c: (match(c.toks, ANSWER + ["key"]), _extra(c.toks, ANSWER))), group
    return None


def _translation(cols: list[Col], texts: list[Col], language: str | None, description: str) -> tuple[str, str] | None:
    """Параллельные тексты (translation.en/translation.ru, text_en/text_ru, en/ru) → (вход, ответ).
    Направление — по постановке («на английский»), иначе ответ на языке задачи."""
    coded = {}
    for c in texts:
        code = lang_code(c.name)
        if code and code not in coded and sum(t in _ALIAS for t in c.toks) == 1:
            coded[code] = c
    if len(coded) < 2:
        return None
    bases = {tuple(t for t in c.toks if t not in _ALIAS) for c in coded.values()}
    others = [c for c in texts if c not in coded.values()]
    if not re.search(r"перев|translat", description.lower()) and (len(bases) > 1 or others):
        return None
    tgt = target_language(description) or language
    tgt = tgt if tgt in coded else list(coded)[1]
    src = next((x for x in ([language] if language != tgt else []) + ["en"] + list(coded) if x in coded and x != tgt))
    return coded[src].name, coded[tgt].name


def _classification(cols, df, label_names, spec_labels, language) -> dict:
    spec_norm = {normalize_label(lab) for lab in spec_labels or []}
    cands = []
    for i, c in enumerate(cols):
        multi = c.kind in ("str_list", "int_list") and sum(len(list(v)) == 1 for v in c.values) >= 0.5 * len(c.values)
        if not (c.kind == "category" or c.name in label_names or multi):
            continue
        vals = [x for x in (label_value(v) for v in c.values) if x]
        if not 2 <= len(set(vals)) <= MAX_CLASSES:
            continue
        rank = match(c.toks, TARGET_CLS)
        overlap = len({normalize_label(v) for v in vals} & spec_norm) / len(spec_norm) if spec_norm else 0.0
        letters = any(re.search(r"[^\W\d_]", v) for v in vals)  # «Bad/Good» понятнее, чем 0/2
        cands.append(((-round(overlap, 2), rank is None, c.name not in label_names, rank or 0, c.meta, not letters,
                       _extra(c.toks, TARGET_CLS), i), c))
    target = min(cands, key=lambda x: x[0])[1] if cands else None
    exclude = {target.name} if target else set()
    inputs = _inputs(cols, df, exclude, language, "classification")
    return {"input_column": inputs[0] if inputs else None, "extra_input_columns": inputs[1:],
            "target_column": target.name if target else None}


def _generation(cols, df, language, description, task_key) -> dict:
    chats = [c for c in cols if c.kind == "messages"]
    if chats:
        c = min(chats, key=lambda c: (match(c.toks, CHAT) is None, cols.index(c)))
        return {"kind": "chat", "input_column": c.name, "target_column": c.name}
    mc = _multiple_choice(cols)
    if mc:
        key, group = mc
        inputs = _inputs(cols, df, {key.name, *group}, language, "generation")
        return {"input_column": inputs[0] if inputs else None, "extra_input_columns": inputs[1:],
                "target_column": key.name, "options_columns": group}
    texts = [c for c in cols if (c.is_text or phrases(c)) and c.fill >= 0.05]
    pair = _translation(cols, texts, language, description)
    if pair:
        return {"input_column": pair[0], "target_column": pair[1]}
    vocab = TARGET_SUM if task_key == "summarization" else TARGET_GEN
    named = [c for c in texts if match(c.toks, vocab) is not None]
    if not named:  # ответы-структуры: SQuAD {text: [...]}, список эталонов
        named = [c for c in cols if c.kind in ("dict", "str_list", "dict_list") and not c.meta
                 and match(c.toks, ANSWER) is not None]
    target = min(named, key=lambda c: (match(c.toks, vocab) is None, match(c.toks, vocab) or 0,
                                       _extra(c.toks, vocab), not c.lang_ok(language), cols.index(c))) if named else None
    if target is None and len(texts) >= 2:  # имена не подсказали
        if task_key == "summarization":  # ответ — заметно короче входа
            longest = max(texts, key=lambda c: c.mean_len)
            shorter = [c for c in texts if c.mean_len < 0.6 * longest.mean_len]
            target = max(shorter, key=lambda c: c.mean_len) if shorter else None
        else:
            main = _inputs(cols, df, set(), language, "generation")[0]
            rest = [c for c in texts if c.name != main]
            target = min(rest, key=lambda c: (not c.lang_ok(language), cols.index(c))) if rest else None
    exclude = {target.name} if target else set()
    inputs = _inputs(cols, df, exclude, language, "generation", target, vocab + ANSWER)
    return {"input_column": inputs[0] if inputs else None, "extra_input_columns": inputs[1:],
            "target_column": target.name if target else None}


def _qa(cols, df) -> dict:
    mc = _multiple_choice(cols)
    group = mc[1] if mc else []
    used = set(group)
    if mc:
        answer = mc[0]
    else:
        cands = [c for c in cols if c.name not in used and not c.meta and match(c.toks, ANSWER) is not None
                 and (c.is_text or c.kind in ("category", "str_list", "dict", "dict_list", "empty_list"))]
        answer = min(cands, key=lambda c: (match(c.toks, ANSWER), _extra(c.toks, ANSWER), cols.index(c))) if cands else None
    if answer:
        used.add(answer.name)
    texts = [c for c in cols if (c.is_text or phrases(c)) and c.name not in used and c.fill >= 0.5]
    qs = [c for c in texts if match(c.toks, QA_QUESTION) is not None]
    question = min(qs, key=lambda c: (match(c.toks, QA_QUESTION), _extra(c.toks, QA_QUESTION), cols.index(c))) if qs else None
    rest = [c for c in texts if c is not question]
    ctx = [c for c in rest if match(c.toks, QA_CONTEXT) is not None]
    context = min(ctx, key=lambda c: (match(c.toks, QA_CONTEXT), -c.mean_len)) if ctx else None
    if context is None and rest and question:  # без имени: заметно длиннее вопроса
        longest = max(rest, key=lambda c: c.mean_len)
        context = longest if longest.mean_len >= 3 * question.mean_len else None
    return {"kind": "qa", "input_column": context.name if context else None,
            "question_column": question.name if question else None,
            "target_column": answer.name if answer else None, "options_columns": group}


def _ner(cols, df, notes: list[str]) -> dict:
    seqs = [c for c in cols if c.kind == "str_list" and not c.meta]
    tags = [c for c in seqs if _tag_like(c)]
    toks = [c for c in seqs if c not in tags and not _subword(c)]

    def aligned(a: Col, b: Col) -> bool:
        pairs = list(zip(df[a.name].head(PROBE_ROWS), df[b.name].head(PROBE_ROWS)))
        return sum(_is_seq(x) and _is_seq(y) and len(x) == len(y) for x, y in pairs) >= 0.9 * len(pairs)
    for tag in sorted(tags, key=lambda c: (match(c.toks, TAGS) is None, match(c.toks, TAGS) or 0)):
        for tok in sorted(toks, key=lambda c: (match(c.toks, TOKENS) is None, match(c.toks, TOKENS) or 0)):
            if aligned(tok, tag):
                return {"kind": "ner_bio", "input_column": tok.name, "target_column": tag.name}
    spans = [c for c in cols if c.kind == "spans"]
    texts = [c for c in cols if c.strings and c.kind in ("text", "category", "constant") and not c.meta]
    if spans and texts:
        sp = min(spans, key=lambda c: (match(c.toks, SPANS) is None, cols.index(c)))

        def fits(t: Col) -> float:  # позиции и тексты сущностей сходятся с этой колонкой
            ok = total = 0
            for text, ents in zip(df[t.name].head(50), df[sp.name].head(50)):
                for e in list(parsed(ents)) if _is_seq(parsed(ents)) else []:
                    if isinstance(e, dict) and isinstance(e.get("start"), (int, np.integer)):
                        total += 1
                        ent = next((_as_text(e[k]) for k in SPAN_TEXT_KEYS if k in e), None)
                        ok += ent is None or _as_text(text)[int(e["start"]):int(e.get("end", 0))] == ent
            return ok / total if total else 0.0
        text = min(texts, key=lambda c: (-fits(c), match(c.toks, INPUT) is None, match(c.toks, INPUT) or 0, -c.mean_len))
        return {"kind": "ner_spans", "input_column": text.name, "target_column": sp.name}
    if any(c.kind == "int_list" and match(c.toks, TAGS) is not None for c in cols):
        notes.append("теги сущностей — числа без названий классов")
    return {"kind": "ner_bio", "input_column": toks[0].name if toks else None, "target_column": None}


def infer_schema(df: pd.DataFrame, fmt: str, family: TaskFamily, label_names: dict | None = None,
                 spec_labels: list[str] | None = None, language: str | None = None, description: str = "",
                 task_key: str = "") -> dict:
    """Схема колонок под формат задачи: поля MapSchemaOp (kind, input_column, target_column, question_column,
    extra_input_columns, options_columns, language_filter) и notes — почему чего-то не нашлось.
    None в колонке — не найдена."""
    df = flatten_dicts(df)
    cols = profile_columns(df)
    flt = _language_filter(cols, language)
    if flt:
        part = filter_language(df, flt)
        if len(part):  # роли — по строкам нужного языка
            df, cols = part, profile_columns(part)
    schema = {"kind": "plain", "input_column": None, "target_column": None, "question_column": None,
              "extra_input_columns": [], "options_columns": [], "language_filter": flt, "notes": []}
    if fmt == "entities":
        schema.update(_ner(cols, df, schema["notes"]))
    elif fmt == "span":
        schema.update(_qa(cols, df))
    elif family == "classification":
        schema.update(_classification(cols, df, label_names or {}, spec_labels, language))
    else:
        schema.update(_generation(cols, df, language, description, task_key))
    return schema


# ---------------------------------------------------------------- приведение к формату задачи

def qa_input(context: str, question: str) -> str:
    return f"Контекст: {context}\nВопрос: {question}"


def row_id(inp: str, tgt: str) -> str:
    return hashlib.sha1(f"{inp}\x1f{tgt}".encode("utf-8")).hexdigest()[:16]


def _text_target(v) -> str:
    """Ответ генерации: словарь ответов (SQuAD, DROP — в том числе строкой) → первый ответ; список эталонов → первый;
    прочие строки — как есть (ответом может быть и JSON)."""
    p = parsed(v)
    if isinstance(p, dict) and {"text", "answer", "spans", "number"} & set(p) or (not isinstance(v, str) and _is_seq(p)):
        a = answers(p)
        return a[0] if a else ""
    return _as_text(v)


def schema_columns(op: MapSchemaOp) -> list[str]:
    cols = [op.input_column, op.target_column, *op.extra_input_columns, *op.options_columns, *op.language_filter]
    return list(dict.fromkeys(cols + ([op.question_column] if op.kind == "qa" else [])))


def to_task_frame(df: pd.DataFrame, op: MapSchemaOp, fmt: str | None = None) -> pd.DataFrame:
    """Приводит датасет к формату задачи: input/target строками (цель — одной строкой), _id, synthetic=False.
    Для QA ещё колонка alt — все эталонные ответы (json). fmt="label" — у классификации из списка меток
    берётся единственная, строки с несколькими метками получают пустую цель (отбрасываются при разбиении).
    Строки без входа или ответа остаются — их отбрасывает drop_unusable."""
    df = flatten_dicts(df)
    missing = [c for c in schema_columns(op) if not c or c not in df.columns]
    if missing:
        raise ValueError(f"В датасете нет колонок {missing}; есть {list(df.columns)}")
    if op.language_filter:
        df = filter_language(df, op.language_filter)
    df = df.reset_index(drop=True)
    options = options_per_row(df, op.options_columns) if op.options_columns else None
    chosen = resolve_choices(df[op.target_column], options) if options else None
    alt = None
    if op.kind == "ner_bio":
        inputs = [detokenize(t) if _is_seq(t) else _as_text(t) for t in df[op.input_column]]
        targets = [format_entities(bio_to_entities(t, g)) if _is_seq(t) and _is_seq(g) else NO_ENTITIES
                   for t, g in zip(df[op.input_column], df[op.target_column])]
    elif op.kind == "ner_spans":
        inputs = [_as_text(t) for t in df[op.input_column]]
        targets = [format_entities(spans_to_entities(t, s)) for t, s in zip(inputs, df[op.target_column])]
    elif op.kind == "chat":
        inputs, targets = map(list, zip(*(chat_pair(v) for v in df[op.input_column]))) if len(df) else ([], [])
    elif op.kind == "qa":
        if options:
            golds = [[a] if a else [] for a in chosen]
        else:
            golds = [answers(v) for v in df[op.target_column]]
        inputs = [qa_input(_as_text(c).strip(), _as_text(q).strip()) for c, q in zip(df[op.input_column], df[op.question_column])]
        targets, alt = [a[0] if a else "" for a in golds], [json.dumps(a, ensure_ascii=False) for a in golds]
    else:
        inputs = render_inputs(df, [op.input_column, *op.extra_input_columns])
        if options:
            targets = chosen
        elif fmt == "label":
            targets = [label_value(v) for v in df[op.target_column]]
        else:
            targets = [_text_target(v) for v in df[op.target_column]]
    if options:
        inputs = [f"{i}\n{options_block(t)}" if t else i for i, (t, _) in zip(inputs, options)]
    out = pd.DataFrame({"input": inputs, "target": targets})
    if alt is not None:
        out["alt"] = alt
    if op.label_map:
        out["target"] = out["target"].map(lambda v: op.label_map.get(v, v))
    out["input"] = out["input"].astype(str).str.strip()
    # ответ модели читается до первого перевода строки — эталон тоже держим одной строкой
    out["target"] = out["target"].astype(str).str.replace(r"\s*\n+\s*", " ", regex=True).str.strip()
    out["synthetic"] = False
    out.insert(0, "_id", [row_id(i, t) for i, t in zip(out["input"], out["target"])])
    return out.reset_index(drop=True)


def drop_unusable(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Строки без входа или без ответа (неразмеченный test-сплит с меткой -1, вопрос без ответа, несколько меток
    у классификации) непригодны ни для обучения, ни для оценки — убираются до разбиения на train/val/test."""
    ok = (df["input"].str.strip() != "") & (df["target"].str.strip() != "")
    return df[ok].reset_index(drop=True), int((~ok).sum())
