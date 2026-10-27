"""Файловое хранилище (§5): атомарная запись, jsonl-журналы, блокировки сессий и GPU."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from pydantic import BaseModel

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                      wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _INVALID_HANDLE = ctypes.c_void_p(-1).value
else:
    import fcntl

M = TypeVar("M", bound=BaseModel)
T = TypeVar("T")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Пишет во временный файл в той же папке и подменяет целевой через os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _retry_busy(lambda: os.replace(tmp, path))
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _retry_busy(fn: Callable[[], T]) -> T:
    """На Windows подмена файла, открытого другим процессом, и чтение файла в момент подмены дают PermissionError
    (так же мешает антивирус) — повторяем, пока файл не освободится."""
    deadline = time.monotonic() + 5
    while True:
        try:
            return fn()
        except PermissionError:
            if os.name != "nt" or time.monotonic() > deadline:
                raise
            time.sleep(0.01)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: Path, obj: Any) -> None:
    if isinstance(obj, BaseModel):
        text = obj.model_dump_json(indent=2)
    else:
        text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    atomic_write_text(path, text)


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(_retry_busy(lambda: path.read_text(encoding="utf-8")))


def read_model(path: Path, model: type[M]) -> M | None:
    path = Path(path)
    if not path.exists():
        return None
    return model.model_validate_json(_retry_busy(lambda: path.read_text(encoding="utf-8")))


def append_jsonl(path: Path, obj: Any) -> None:
    """Дописывает одну строку; одна запись write() < PIPE_BUF не перемешивается между процессами."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = obj.model_dump_json() if isinstance(obj, BaseModel) else json.dumps(obj, ensure_ascii=False, default=str)
    if os.name == "nt":
        _append_windows(path, (line + "\n").encode("utf-8"))
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _append_windows(path: Path, data: bytes) -> None:
    """Режим "a" в Windows CRT — это seek в конец + write: строки разных процессов затирают друг друга. Дескриптор
    только с FILE_APPEND_DATA ядро дописывает в конец атомарно."""
    h = _kernel32.CreateFileW(str(path), 0x4, 0x7, None, 4, 0x80, None)  # FILE_APPEND_DATA, SHARE_*, OPEN_ALWAYS
    if h is None or h == _INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    fd = msvcrt.open_osfhandle(h, 0)  # без O_TEXT — двоичный режим, \n не превращается в \r\n
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def read_jsonl(path: Path, offset: int = 0) -> tuple[list[dict], int]:
    """Читает строки, начиная с байтового смещения. Возвращает (записи, новое смещение).

    Незавершённая последняя строка (процесс ещё пишет) не читается и не сдвигает смещение.
    """
    path = Path(path)
    if not path.exists():
        return [], offset
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    records, consumed = [], 0
    for raw in chunk.splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            break
        consumed += len(raw)
        if raw.strip():
            records.append(json.loads(raw))
    return records, offset + consumed


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


_session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def session_lock(session_id: str) -> asyncio.Lock:
    """Один asyncio.Lock на сессию: обработка запросов сессии идёт строго по очереди."""
    return _session_locks[session_id]


@contextmanager
def file_lock(path: Path, blocking: bool = True) -> Iterator[bool]:
    """Межпроцессная блокировка: flock, на Windows — msvcrt.locking первого байта. Отдаёт True, если замок взят."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    acquired = False
    try:
        acquired = _lock(f, blocking)
        yield acquired
    finally:
        try:
            if acquired and not f.closed:
                _unlock(f)
        finally:
            f.close()


def _lock(f, blocking: bool) -> bool:
    if os.name != "nt":
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            return True
        except BlockingIOError:
            return False
    f.seek(0)  # msvcrt.locking берёт байты от текущей позиции; LK_LOCK сдаётся через 10 с — ждём сами
    while True:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except PermissionError:
            if not blocking:
                return False
            time.sleep(0.2)


def _unlock(f) -> None:
    if os.name != "nt":
        fcntl.flock(f, fcntl.LOCK_UN)
        return
    f.seek(0)
    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
