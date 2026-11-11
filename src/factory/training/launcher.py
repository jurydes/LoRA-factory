"""Запуск рабочего процесса обучения отдельным процессом (новая сессия ОС — переживает закрытие вкладки)."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from factory.core import activity
from factory.core.storage import read_json
from factory.runs import RunPaths

SRC_DIR = Path(__file__).resolve().parents[2]
# Ctrl+C и закрытие консоли сервиса не должны доходить до обучения: на POSIX — новая сессия, на Windows — своя
# группа процессов и своя (скрытая) консоль
DETACHED = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt"
            else {"start_new_session": True})


def spawn(paths: RunPaths, mode: str, status_file: Path, *extra: str) -> subprocess.Popen:
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.unlink(missing_ok=True)
    env = os.environ.copy()
    from factory.core.config import get_config
    offline = get_config().offline
    env["FACTORY_OFFLINE"] = "1" if offline else "0"
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if offline:
            env[key] = "1"
        else:
            env.pop(key, None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC_DIR), env.get("PYTHONPATH")]))
    env.setdefault("PYTHONUTF8", "1")  # лог пишется в UTF-8; на Windows stdout иначе в cp1251 и падает на эмодзи
    cmd = [sys.executable, "-m", "factory.training.worker", "--run-dir", str(paths.root), "--mode", mode,
           "--status-file", str(status_file), *extra]
    log = open(paths.train_log, "a", encoding="utf-8")
    log.write(f"\n===== {mode} {' '.join(extra)} =====\n")
    log.flush()
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(SRC_DIR), **DETACHED)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":  # os.kill(pid, 0) на Windows — это CTRL_C_EVENT, а не проверка
        import psutil
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # зомби (завершился, но не собран) считается мёртвым
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return True


async def wait(proc: subprocess.Popen, status_file: Path, poll_s: float = 1.0,
               on_progress: Callable[[str], Awaitable[None]] | None = None) -> dict:
    """Ждёт процесс; ход работы из файла статуса (поле progress) передаёт в on_progress."""
    shown = None
    while proc.poll() is None:
        await asyncio.sleep(poll_s)
        if on_progress:
            progress = (read_json(status_file, {}) or {}).get("progress")
            if progress and progress != shown:
                shown = progress
                await on_progress(progress)
    return read_json(status_file) or {"status": "error", "message": f"процесс завершился с кодом {proc.returncode} без статуса"}


MODES = {"baseline": "считает baseline'ы на val: загрузка модели, ответы исходной модели",
         "pilot": "пробный прогон: скорость, память, оценка времени обучения",
         "final_eval": "финальная оценка на test: ответы адаптера и исходной модели, метрики"}


async def run(paths: RunPaths, mode: str, status_file: Path, *extra: str) -> dict:
    key = f"worker:{mode}:{time.time()}"
    await activity.emit(MODES.get(mode, mode), "рабочий процесс", key=key)

    async def progress(text: str) -> None:  # итог строки ленты обновляется по ходу работы
        await activity.emit(text, kind="result", key=key)
    return await wait(spawn(paths, mode, status_file, *extra), status_file, on_progress=progress)
