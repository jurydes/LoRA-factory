"""Разбиение train/val/test. test сразу закрывается: читается только при окончательной остановке."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from factory.core.contracts import SplitProportions
from factory.core.storage import append_jsonl
from factory.prep.ops import normalize_text

logger = logging.getLogger(__name__)

TEST_FILE = "test.sealed.parquet"


class SealedTestError(PermissionError):
    pass


def split_frame(df: pd.DataFrame, proportions: SplitProportions, seed: int,
                stratify: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Группы одинаковых входов целиком попадают в один сплит — без утечки дублей в test.

    При stratify распределение меток сохраняется: группы раскладываются внутри каждой метки.
    """
    rng = np.random.default_rng(seed)
    groups = df["input"].map(normalize_text)
    assignment = pd.Series("train", index=df.index)

    strata = df.groupby(df["target"] if stratify else pd.Series(0, index=df.index), sort=True)
    for _, part in strata:
        keys = groups.loc[part.index].unique()
        keys = keys[rng.permutation(len(keys))]
        n = len(keys)
        n_test = int(round(n * proportions.test))
        n_val = int(round(n * proportions.val))
        if n >= 3:
            n_test, n_val = max(n_test, 1 if proportions.test > 0 else 0), max(n_val, 1 if proportions.val > 0 else 0)
        test_keys, val_keys = set(keys[:n_test]), set(keys[n_test:n_test + n_val])
        part_groups = groups.loc[part.index]
        assignment.loc[part.index[part_groups.isin(test_keys).to_numpy()]] = "test"
        assignment.loc[part.index[part_groups.isin(val_keys).to_numpy()]] = "val"

    out = [df[assignment == s].reset_index(drop=True) for s in ("train", "val", "test")]
    return out[0], out[1], out[2]


def frame_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for i, t in sorted(zip(df["input"], df["target"])):
        h.update(f"{i}\x1f{t}\x1e".encode("utf-8"))
    return h.hexdigest()


def save_splits(data_dir: Path, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> str:
    """Сохраняет сплиты; возвращает test_hash."""
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(data_dir / "train.parquet", index=False)
    val.to_parquet(data_dir / "val.parquet", index=False)
    test.to_parquet(data_dir / TEST_FILE, index=False)
    return frame_hash(test)


def read_test(data_dir: Path, *, final: bool, expected_hash: str, events_path: Path | None = None) -> pd.DataFrame:
    """Единственный путь к test. Без final=True (окончательная остановка обучения) — отказ."""
    if not final:
        raise SealedTestError("test закрыт до окончательной остановки обучения")
    df = pd.read_parquet(data_dir / TEST_FILE)
    if frame_hash(df) != expected_hash:
        raise SealedTestError("хэш test не совпадает с манифестом — test изменён")
    if events_path is not None:
        append_jsonl(events_path, {"type": "test_opened", "rows": len(df)})
    return df
