"""Точка подключения будущего объединения нескольких пользовательских датасетов."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class MergeRequest(BaseModel):
    sources: list[Path] = Field(min_length=2)
    column_maps: list[dict[str, str]] = Field(default_factory=list)
    label_maps: list[dict[str, str]] = Field(default_factory=list)


def merge_datasets(request: MergeRequest) -> None:
    """Заглушка: контракт есть, но объединение пока не выдаёт фиктивный датасет."""
    raise NotImplementedError("Объединение датасетов пока не реализовано")
