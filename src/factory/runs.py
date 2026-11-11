"""Реестр адаптеров и запусков: папки runs/{timestamp}_{model}_{task}_{short_id}/ и манифесты (§5)."""
from __future__ import annotations

import importlib.metadata
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from factory.core.contracts import DatasetCandidate, Manifest, Recipe, RunConfig, TaskSpec
from factory.core.storage import read_model, stable_hash, write_json

LIBRARIES = ["torch", "transformers", "trl", "peft", "datasets", "pydantic-ai-slim", "sentence-transformers",
             "cleanlab", "datasketch", "scikit-learn", "mlflow", "unsloth", "bitsandbytes"]


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def run_id(self) -> str:
        return self.root.name

    manifest = property(lambda self: self.root / "manifest.json")
    config = property(lambda self: self.root / "config.json")
    recipe = property(lambda self: self.root / "recipe.json")
    eval_report = property(lambda self: self.root / "eval_report.json")
    data = property(lambda self: self.root / "data")
    metrics = property(lambda self: self.root / "metrics.jsonl")
    events = property(lambda self: self.root / "events.jsonl")
    control = property(lambda self: self.root / "control.json")
    heartbeat = property(lambda self: self.root / "heartbeat")
    adapter = property(lambda self: self.root / "adapter")
    llm_calls = property(lambda self: self.root / "llm_calls.jsonl")
    train_log = property(lambda self: self.root / "train.log")
    pilot = property(lambda self: self.root / "pilot")
    baselines_val = property(lambda self: self.root / "baselines_val.json")

    def round_dir(self, k: int) -> Path:
        return self.root / "rounds" / f"round_{k}"


def library_versions() -> dict[str, str]:
    out = {}
    for lib in LIBRARIES:
        try:
            out[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def reuse_key(spec: TaskSpec, dataset: DatasetCandidate, recipe: Recipe, config: RunConfig) -> str:
    """Хэш постановки + датасета + рецепта + конфигурации: совпадение — можно взять готовый адаптер."""
    model_dir = Path(spec.base_model_id)
    local_files = None
    if model_dir.is_dir():
        relevant = [model_dir / "config.json", *model_dir.glob("*.safetensors"), *model_dir.glob("pytorch_model*.bin")]
        local_files = sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in relevant if p.is_file())
    return stable_hash({
        "spec": spec.model_dump(include={"task_type", "language", "labels", "description", "base_model_id"}),
        "dataset": [dataset.hf_id, dataset.revision],
        "local_model_files": local_files,
        "recipe": recipe.model_dump(mode="json", include={"split_seed", "proportions", "ops"}),
        # seeds не входят: train_seed случаен на каждый запуск, split_seed уже есть в рецепте
        "config": config.model_dump(mode="json", exclude={"rationales", "strategy", "seeds"}),
    })


def new_run_dir(runs_dir: Path, model_id: str, task_type: str) -> RunPaths:
    slug = model_id.replace("\\", "/").rstrip("/").split("/")[-1]  # HF id или локальный путь (на Windows — C:\…)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}_{task_type}_{uuid.uuid4().hex[:6]}"
    root = runs_dir / name
    root.mkdir(parents=True)
    return RunPaths(root)


def create_run(runs_dir: Path, spec: TaskSpec, dataset: DatasetCandidate, recipe: Recipe, config: RunConfig,
               session_dir: Path, session_id: str, restart_of: str | None = None) -> tuple[RunPaths, Manifest]:
    """Создаёт папку запуска, копирует данные сессии (test остаётся закрытым) и пишет манифест."""
    paths = new_run_dir(runs_dir, spec.base_model_id, spec.task_type)
    shutil.copytree(session_dir / "data", paths.data)
    for synth in session_dir.glob("synthetic_*.jsonl"):
        shutil.copy2(synth, paths.root / synth.name)
    write_json(paths.recipe, recipe)
    write_json(paths.config, config)
    n_train = recipe.profile_after.n_rows if recipe.profile_after else 0
    manifest = Manifest(
        run_id=paths.run_id, spec=spec, dataset=dataset, config=config, seeds=config.seeds,
        test_hash=recipe.test_hash, n_train=n_train, library_versions=library_versions(),
        restart_of_run_id=restart_of, session_id=session_id, reuse_key=reuse_key(spec, dataset, recipe, config),
    )
    write_json(paths.manifest, manifest)
    return paths, manifest


def load_manifest(paths: RunPaths) -> Manifest:
    m = read_model(paths.manifest, Manifest)
    if m is None:
        raise FileNotFoundError(paths.manifest)
    return m


def save_manifest(paths: RunPaths, manifest: Manifest) -> None:
    write_json(paths.manifest, manifest)


def find_reusable(runs_dir: Path, key: str) -> Manifest | None:
    from factory.hparams.history import iter_manifests
    for m in iter_manifests(runs_dir):
        if m.reuse_key == key and m.status == "finished" and m.adapter_path:
            return m
    return None
