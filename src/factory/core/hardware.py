"""Характеристики машины для подбора моделей: GPU, VRAM, RAM, диск и замеренная производительность.

Замер — несколько bf16/fp16 matmul на GPU: реальные TFLOPS этой карты, а не паспортные. Результат кэшируется
в DATA_DIR/cache/hardware.json (пересчёт — refresh=True или при смене GPU)."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from factory.core.storage import read_model, write_json

logger = logging.getLogger(__name__)
GB = 1024 ** 3


class Hardware(BaseModel):
    gpu_name: str | None = None
    gpu_count: int = 0
    vram_gb: float = 0.0
    compute_capability: str | None = None
    bf16: bool = False
    measured_tflops: float | None = None  # эффективные TFLOPS на matmul (bf16/fp16)
    ram_gb: float = 0.0
    ram_available_gb: float = 0.0
    disk_free_gb: float = 0.0
    cpu_count: int = 1
    vram_others_gb: float = 0.0  # занято другими программами в момент подбора (vram_used_by_others)

    @property
    def has_gpu(self) -> bool:
        return self.gpu_count > 0 and self.vram_gb > 0

    @property
    def usable_vram_gb(self) -> float:
        """Видеопамять, доступная обучению: перебор на Windows не даёт ошибки, а молча замедляет его в разы."""
        return max(self.vram_gb - self.vram_others_gb, 0.0)


def _ram() -> tuple[float, float]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / GB, vm.available / GB
    except ImportError:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
        return info.get("MemTotal", 0) / GB, info.get("MemAvailable", 0) / GB


def benchmark_tflops(size: int = 4096, iters: int = 20) -> float | None:
    """Эффективные TFLOPS: квадратный matmul в bf16 (или fp16) на первой GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        a = torch.randn(size, size, device="cuda", dtype=dtype)
        b = torch.randn(size, size, device="cuda", dtype=dtype)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        del a, b
        torch.cuda.empty_cache()
        return round(2 * size ** 3 * iters / elapsed / 1e12, 1)
    except Exception as e:
        logger.warning("Замер производительности GPU не удался: %s", e)
        return None


def probe(disk_path: Path) -> Hardware:
    hw = Hardware(cpu_count=os.cpu_count() or 1)
    hw.ram_gb, hw.ram_available_gb = (round(x, 1) for x in _ram())
    disk_path.mkdir(parents=True, exist_ok=True)
    hw.disk_free_gb = round(shutil.disk_usage(disk_path).free / GB, 1)
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            hw.gpu_name, hw.gpu_count = props.name, torch.cuda.device_count()
            hw.vram_gb = round(props.total_memory / GB, 1)
            hw.compute_capability = f"{props.major}.{props.minor}"
            hw.bf16 = torch.cuda.is_bf16_supported()
    except Exception as e:
        logger.warning("GPU не определена: %s", e)
    return hw


_cached: dict[str, Hardware] = {}


def detect(data_dir: Path, refresh: bool = False) -> Hardware:
    """Характеристики машины. GPU опрашивается в отдельном процессе: CUDA-контекст в процессе сервиса занял бы
    VRAM на всё время работы. Замер TFLOPS берётся из кэша, если GPU та же."""
    if str(data_dir) in _cached and not refresh:
        return _cached[str(data_dir)]
    cache = data_dir / "cache" / "hardware.json"
    previous = read_model(cache, Hardware)
    need_bench = refresh or previous is None or previous.measured_tflops is None
    try:
        out = subprocess.run([sys.executable, "-m", "factory.core.hardware", str(data_dir), "1" if need_bench else "0"],
                             capture_output=True, encoding="utf-8", errors="replace", timeout=180, env=_child_env(),
                             cwd=str(Path(__file__).parents[2]))
        hw = Hardware.model_validate_json(out.stdout.strip().splitlines()[-1])
    except Exception as e:
        logger.warning("Опрос железа в подпроцессе не удался (%s) — только CPU/RAM/диск", e)
        hw = Hardware(cpu_count=os.cpu_count() or 1)
        hw.ram_gb, hw.ram_available_gb = (round(x, 1) for x in _ram())
    if not need_bench and previous and previous.gpu_name == hw.gpu_name:
        hw.measured_tflops = previous.measured_tflops
    write_json(cache, hw)
    _cached[str(data_dir)] = hw
    return hw


_others: list[float] = []  # последний замер: пока идёт наше обучение, чужую память от своей не отделить


def _our_worker_running() -> bool:
    try:
        import psutil
        for p in psutil.process_iter(["cmdline"]):
            if "factory.training.worker" in " ".join(p.info.get("cmdline") or []):
                return True
    except Exception:
        pass
    return False


def vram_used_by_others() -> float:
    """ГБ видеопамяти, занятой не нашими процессами (браузеры, редакторы, игры): memory.used из nvidia-smi, пока
    наш рабочий процесс не запущен; во время обучения — последний такой замер (на Windows nvidia-smi не делит память
    по процессам). Без nvidia-smi — 0."""
    if not _our_worker_running():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                                 capture_output=True, encoding="utf-8", timeout=10)
            _others[:] = [round(float(out.stdout.strip().splitlines()[0]) / 1024, 1)]
        except Exception as e:
            logger.debug("nvidia-smi недоступен: %s", e)
    return _others[0] if _others else 0.0


def with_current_usage(hw: Hardware) -> Hardware:
    return hw.model_copy(update={"vram_others_gb": vram_used_by_others()}) if hw.has_gpu else hw


def _child_env() -> dict:
    env = os.environ.copy()
    src = str(Path(__file__).parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
    env.setdefault("PYTHONUTF8", "1")  # stdout читается как UTF-8, на Windows иначе cp1251
    return env


if __name__ == "__main__":  # python -m factory.core.hardware DATA_DIR BENCH(0|1) → JSON в stdout
    hw = probe(Path(sys.argv[1]))
    if hw.has_gpu and sys.argv[2] == "1":
        hw.measured_tflops = benchmark_tflops()
    print(hw.model_dump_json())
