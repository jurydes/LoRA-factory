"""Отдельный каталог чекпоинтов для каждой попытки процесса в раунде."""
from pathlib import Path


def _numbered_dirs(root: Path, prefix: str) -> list[Path]:
    return sorted((p for p in root.glob(f"{prefix}*")
                   if p.is_dir() and p.name[len(prefix):].isdigit()),
                  key=lambda p: int(p.name[len(prefix):]), reverse=True)


def new_attempt(round_dir: Path) -> Path:
    attempts = round_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    previous = _numbered_dirs(attempts, "attempt-")
    number = int(previous[0].name.split("-")[-1]) + 1 if previous else 1
    path = attempts / f"attempt-{number}"
    path.mkdir()
    return path


def last_checkpoint(directory: Path) -> Path | None:
    # Новая попытка может ещё не сохранить ничего: тогда восстанавливаем предыдущую.
    # Корень в конце — совместимость с запусками до введения отдельных попыток.
    for folder in [*_numbered_dirs(directory / "attempts", "attempt-"), directory]:
        checkpoints = _numbered_dirs(folder, "checkpoint-")
        complete = [p for p in checkpoints if (p / "trainer_state.json").exists()]
        if complete:
            if folder == directory:
                # В старых раундах попытки делили каталог, поэтому номер мог начинаться заново.
                return max(complete, key=lambda p: ((p / "trainer_state.json").stat().st_mtime_ns,
                                                     int(p.name.split("-")[-1])))
            return complete[0]
    return None
