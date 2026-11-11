"""Регистрация готового локального каталога модели без обращения к HF Hub."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from factory.core.config import AppConfig
from factory.core.registry import ModelInfo, load_local_models
from factory.core.storage import atomic_write_text, file_lock


def inspect_local_model(path: Path, *, languages: list[str], license: str = "",
                        commercial_ok: bool = False) -> ModelInfo:
    path = path.expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"Модель должна быть каталогом: {path}")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Не найден config.json модели: {path}")
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError(f"Не найдены веса safetensors или pytorch_model.bin: {path}")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    except Exception as e:
        raise ValueError(f"Не удалось прочитать локальный токенизатор {path}: {e}") from e
    from factory.search.models_hub import inspect_config, UNSLOTH_TYPES
    try:
        arch = inspect_config(json.loads(config_path.read_text(encoding="utf-8")))
    except Exception as e:
        raise ValueError(f"Не удалось разобрать архитектуру модели {path}: {e}") from e
    if arch is None:
        raise ValueError("Архитектура модели не поддерживается установленным Transformers/PEFT")
    return ModelInfo(
        id=str(path), params_b=max(arch["params_b"], 1e-6), active_params_b=arch["active_params_b"],
        hidden_size=arch["hidden_size"], num_layers=arch["num_layers"], vocab_size=arch["vocab_size"],
        max_context=arch["max_context"], kv_bytes_per_token=arch.get("kv_bytes_per_token"),
        languages=languages or ["multi"], license=license, commercial_ok=commercial_ok,
        lora_modules=arch["lora_modules"], unsloth=arch["model_type"] in UNSLOTH_TYPES,
        ptuning=True, chat_template=bool(getattr(tok, "chat_template", None)),
        text_only=arch["text_only"], source="local",
    )


def register_local_model(cfg: AppConfig, path: Path, *, languages: list[str], license: str = "",
                         commercial_ok: bool = False) -> ModelInfo:
    resolved = path.expanduser().resolve(strict=True)
    if cfg.models_dir is not None and not resolved.is_relative_to(cfg.models_dir.expanduser().resolve()):
        raise ValueError(f"Модель должна находиться в {cfg.models_dir}")
    model = inspect_local_model(resolved, languages=languages, license=license, commercial_ok=commercial_ok)
    file = cfg.registry_dir / "local_models.yaml"
    with file_lock(cfg.registry_dir / "local_models.lock"):
        existing = [m for m in load_local_models(cfg.registry_dir) if m.id != model.id]
        raw = [m.model_dump(mode="json") for m in [*existing, model]]
        atomic_write_text(file, yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
    return model
