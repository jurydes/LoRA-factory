import logging
import numpy as np
from pathlib import Path
from datasets import load_dataset, DatasetDict, concatenate_datasets
from generation import generate_task, AsyncList
import fasttext
from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger(__name__)

LANGDETECT_MODEL_DIR = str(Path(__file__).parent / "lid.176.bin")
langdetect_model = fasttext.load_model(LANGDETECT_MODEL_DIR)


def detect_language(text: str, block_size: int = 500) -> tuple[str, float]:
    """Определяет язык текста по блокам. Возвращает (код языка, вероятность)."""
    lang_dict = {}
    for i in range(1, int(np.ceil(len(text) / block_size)) + 1):
        block = " ".join(text[(i-1)*block_size : i*block_size].split())
        lang, prob = list(zip(*langdetect_model.predict(block)))[0]
        lang = lang.replace("__label__", "")
        lang_dict.setdefault(lang, []).append(prob)

    langs = [(sum(probs) / len(lang_dict), lang) for lang, probs in lang_dict.items()]
    prob, lang = langs[np.argmax(langs, axis=0)[0]]
    return lang, prob


def count_russian(dataset, col: str = "input", sample: int = 500) -> int:
    """Оценивает количество русскоязычных примеров в датасете по выборке до sample строк."""
    indices = list(range(min(sample, len(dataset))))
    texts = dataset.select(indices)[col]
    russian = sum(1 for t in texts if detect_language(str(t))[0] == "ru")
    if len(dataset) > sample:
        russian = int(russian * len(dataset) / sample)
    return russian


def print_dataset_readme(dataset_id: str, n_lines: int = 5) -> None:
    """Печатает первые n_lines строк README датасета с HuggingFace."""
    try:
        readme_path = hf_hub_download(repo_id=dataset_id, filename="README.md", repo_type="dataset")
        with open(readme_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        print("\n".join(lines[:n_lines]))
        if len(lines) > n_lines:
            print("... (полный README по ссылке выше)")
    except Exception as e:
        print(f"  README недоступен: {e}")


def search_datasets(config) -> list[dict]:
    """Ищет русскоязычные датасеты на HuggingFace Hub по параметрам конфига."""
    api = HfApi()
    by_language = api.list_datasets(language="ru", limit=config.target_count, full=True)

    seen = {}
    for ds in list(by_language):
        if ds.description and ds.id not in seen:
            seen[ds.id] = {
                "id": ds.id,
                "description": (ds.description or "")[:1000],
                "tags": ds.tags or [],
                "score": 0,
            }
    return list(seen.values())


SCORING_PROMPT = """
Запрос пользователя: {query}
Датасет: {id}
Описание: {description}
Теги: {tags}

## Задача
Определи насколько этот датасет подходит под запрос. Чем больше датасет решает именно эту задачу, тем больше он подходит под нее.
Цени больше ориентированность под конкретную задачу, чем общий объем. Оценка - оценка того, на сколько процентов датасет подходит под запрос пользователя.

## Шкала:
100 должна быть оценкой которую очень трудно получить - только если датасет идеально создан именно под конкретную специфическую задачу.
Выше 90 - идеальный датасет.
Выше 80 - очень хороший датасет который совсем немного не дотянул до идеального.
Выше 70 - хороший датасет с небольшими минусами.
Выше 60 - хороший датасет с весомым минусом.
Выше 50 - средний датасет.
Ниже 50 - на твое усмотрение.
Просто "общие датасеты", а также датасеты для претрейна оцениваются в 0.
Если нет какого-то разделения на input/target ПО ОТДЕЛЬНЫМ КОЛОНКАМ - ставь 0

## Формат ответа:
Ответь одной цифрой от 0 до 100, без доп. объяснений.
"""


async def score_datasets(query: str, datasets: list[dict]) -> list[dict]:
    """Оценивает релевантность датасетов запросу через LLM. Возвращает список, отсортированный по убыванию score."""
    tasks = AsyncList()
    for ds in datasets:
        tasks.append(generate_task(SCORING_PROMPT.format(**ds, query=query), temperature=0.0, max_tokens=5))

    await tasks.complete_couroutines(batch_size=20, verbose=True)

    for ds, answer in zip(datasets, tasks.contents):
        try:
            ds["score"] = float(str(answer).strip())
        except ValueError:
            ds["score"] = 0.0

    return sorted(datasets, key=lambda ds: ds["score"], reverse=True)


async def check_available_datasets(query: str, config) -> list[dict]:
    """Ищет и ранжирует датасеты на HuggingFace по описанию задачи. Возвращает топ-10 подходящих."""
    print("\nШаг 1: ищем все датасеты с русским языком на HuggingFace...")
    datasets = search_datasets(config)
    print(f"Найдено: {len(datasets)}")

    print("\nШаг 2: LLM оценивает релевантность...")
    datasets = await score_datasets(query, datasets)
    datasets = [ds for ds in datasets if ds["score"]][:10]

    print(f"\nГотово. Подходящих датасетов: {len(datasets)}")
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"https://huggingface.co/datasets/{ds['id']}  (score: {ds['score']})")
        print_dataset_readme(ds['id'])

    return datasets


async def detect_columns(dataset_name: str, column_names: list[str]) -> tuple[str | None, str | None]:
    """Определяет через LLM, какая колонка датасета является входом, а какая — целевым ответом."""
    prompt = f"""
Датасет: {dataset_name}
Колонки: {column_names}

Определи какая колонка является входом (input) для модели, а какая целевым ответом (target).
Ответь строго в формате:
input: <название колонки>
target: <название колонки>
"""
    answer = await generate_task(prompt, temperature=0.0, max_tokens=50)

    input_col = target_col = None
    for line in answer.strip().splitlines():
        if line.startswith("input:"):
            input_col = line.split(":", 1)[1].strip()
        elif line.startswith("target:"):
            target_col = line.split(":", 1)[1].strip()

    return input_col, target_col


def make_preprocess(tokenizer, max_length):
    def preprocess(batch):
        texts = [
            f"Input: {inp}\nTarget: {tgt}{tokenizer.eos_token}"
            for inp, tgt in zip(batch["input"], batch["target"])
        ]
        encoded = tokenizer(texts, max_length=max_length, truncation=True, padding="max_length")
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded
    return preprocess


def create_dataset(config, tokenizer, datasets_info: list[tuple[str, str, str]]):
    """Загружает и объединяет датасеты, токенизирует для обучения.

    datasets_info: список кортежей (dataset_name, input_col, target_col).
    Возвращает (raw DatasetDict с колонками input/target, токенизированный DatasetDict).
    """
    all_raw_train = []
    all_raw_test = []

    for dataset_name, input_col, target_col in datasets_info:
        dataset = load_dataset(dataset_name, trust_remote_code=True)

        if "validation" in dataset:
            split = {"train": dataset["train"], "test": dataset["validation"]}
        else:
            split = dataset["train"].train_test_split(test_size=config.val_split, seed=42)

        for key, ds in split.items():
            if input_col != "input":
                ds = ds.rename_column(input_col, "input")
            if target_col != "target":
                ds = ds.rename_column(target_col, "target")
            ds = ds.select_columns(["input", "target"])
            (all_raw_train if key == "train" else all_raw_test).append(ds)

    raw = DatasetDict({
        "train": concatenate_datasets(all_raw_train),
        "test": concatenate_datasets(all_raw_test),
    })

    russian_count = count_russian(raw["train"])
    if russian_count < 100:
        logger.warning(
            "Мало русскоязычных примеров в train: ~%d (рекомендуется >= 100)", russian_count
        )
    else:
        logger.info("Русскоязычных примеров в train: ~%d", russian_count)

    logger.info(
        "Датасеты загружены и объединены: %d train, %d test (из %d датасет(ов))",
        len(raw["train"]), len(raw["test"]), len(datasets_info),
    )

    preprocess = make_preprocess(tokenizer, max_length=config.max_length)
    tokenized = raw.map(preprocess, batched=True, remove_columns=["input", "target"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return raw, tokenized
