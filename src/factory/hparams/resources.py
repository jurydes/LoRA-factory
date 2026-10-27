"""Оценка ресурсов: VRAM, размер батча, max_seq_len. Грубая модель памяти; точные числа даёт пилот."""
from __future__ import annotations

import math

from factory.core.registry import ModelInfo

GB = 1024 ** 3
CUDA_OVERHEAD_GB = 0.8
TARGET_EFFECTIVE_BATCH = 16
KV_OVERHEAD = 3  # фактическая память генерации на последовательность ≈ 2,4× теоретического KV-кэша (буферы Unsloth, префилл)
# батч генерации при оценке на val внутри обучения (evaluator.generate его не превышает) — под него в план раунда
# заложен KV-кэш. Замер (Qwen3-1.7B, RTX 3060): батч 2 — 17 ток/с, 8 — 56, 16 — 60; 4 — компромисс: оценка
# ~в 1,7 раза дольше, чем при 8, зато 8B в QLoRA остаётся доступной на картах 8 ГБ
EVAL_GEN_BATCH = 4


def kv_bytes_per_token(model: ModelInfo) -> int:
    """KV-кэш на токен: из реестра/каталога; иначе K и V на все слои с hidden/4 (типичная GQA), bf16."""
    return model.kv_bytes_per_token or 2 * model.num_layers * (model.hidden_size // 4) * 2


def estimate_memory(model: ModelInfo, seq_len: int, micro_batch: int, quantization: str,
                    gradient_checkpointing: bool, trainable_share: float = 0.005) -> dict[str, float]:
    """Пик памяти раунда: шаг обучения + генерация при оценке на val. Они складываются, а не сменяют друг друга:
    после оценки память генерации остаётся занятой и на следующем шаге обучения (замер: Qwen3-4B, батч 16, промпты
    ~900 токенов — +1,8 ГБ к пику обучения, OOM при пределе 10,3 ГБ)."""
    params = model.params_b * 1e9
    weights = params * (0.55 if quantization == "4bit" else 2.0) / GB
    # активации на слой на токен: ~34·h байт в bf16 без чекпоинтинга, ~2·h с ним (порядок величины)
    per_token_layer = (2 if gradient_checkpointing else 34) * model.hidden_size
    activations = seq_len * micro_batch * model.num_layers * per_token_layer / GB
    logits = seq_len * micro_batch * model.vocab_size * 4 * 2 / GB  # fp32 логиты + градиент
    optimizer = params * trainable_share * 16 / GB  # веса+град+2 момента Adam для обучаемой доли (LoRA r=16 all_linear ≈ 0.5%)
    eval_kv = EVAL_GEN_BATCH * seq_len * kv_bytes_per_token(model) * KV_OVERHEAD / GB
    total = weights + activations + logits + optimizer + eval_kv + CUDA_OVERHEAD_GB
    return {k: round(v, 2) for k, v in dict(weights_gb=weights, activations_gb=activations, logits_gb=logits,
                                            optimizer_gb=optimizer, eval_kv_gb=eval_kv, total_gb=total).items()}


def choose_max_seq_len(prompt_tokens_p99: float, target_tokens_p99: float, model: ModelInfo, cap: int) -> int:
    """По квантилям длин профиля (вход+ответ+шаблон), с округлением до 64 и не больше лимита модели."""
    need = prompt_tokens_p99 + target_tokens_p99 + 48
    return int(min(model.max_context, cap, max(128, math.ceil(need / 64) * 64)))


def choose_batch(model: ModelInfo, seq_len: int, vram_gb: float, safety: float, quantization: str,
                 gradient_checkpointing: bool) -> tuple[int, int] | None:
    """Наибольший micro-batch из 16..1, который помещается; накопление добирает эффективный батч."""
    if vram_gb <= 0:  # CPU: минимальный батч
        return 1, TARGET_EFFECTIVE_BATCH
    for mb in (16, 8, 4, 2, 1):
        if estimate_memory(model, seq_len, mb, quantization, gradient_checkpointing)["total_gb"] <= vram_gb * safety:
            return mb, max(1, TARGET_EFFECTIVE_BATCH // mb)
    return None


def plan_memory(model: ModelInfo, seq_len: int, vram_gb: float, safety: float,
                forced_quantization: str | None) -> dict:
    """Квантизация по VRAM (без неё, если помещается; иначе 4-bit; явный выбор пользователя — как выбрал),
    затем батч; при нехватке — gradient checkpointing."""
    if model.prequantized:  # веса в репозитории уже в 4 битах
        forced_quantization = "4bit"
    quant_options = [forced_quantization] if forced_quantization else ["none", "4bit"]
    for quant in quant_options:
        for ckpt in (False, True):
            batch = choose_batch(model, seq_len, vram_gb, safety, quant, ckpt)
            if batch:
                est = estimate_memory(model, seq_len, batch[0], quant, ckpt)
                return {"quantization": quant, "gradient_checkpointing": ckpt, "micro_batch": batch[0],
                        "grad_accum": batch[1], "estimate": est, "fits": True}
    quant = quant_options[-1]
    est = estimate_memory(model, seq_len, 1, quant, True)
    return {"quantization": quant, "gradient_checkpointing": True, "micro_batch": 1,
            "grad_accum": TARGET_EFFECTIVE_BATCH, "estimate": est, "fits": False}


def steps_per_epoch(n_train: int, micro_batch: int, grad_accum: int) -> int:
    return max(1, math.ceil(n_train / (micro_batch * grad_accum)))


def logging_and_eval_steps(n_train: int, micro_batch: int, grad_accum: int) -> tuple[int, int]:
    """По умолчанию 50 и 400 шагов; для малых данных — не реже двух раз за эпоху."""
    spe = steps_per_epoch(n_train, micro_batch, grad_accum)
    eval_steps = min(400, max(1, spe // 2))
    logging_steps = min(50, max(1, eval_steps // 4))
    return logging_steps, eval_steps


# ---------------------------------------------------------------- пригодность моделей для машины

MFU = 0.35             # доля замеренной производительности matmul, реально достижимая при обучении адаптера
QLORA_SLOWDOWN = 0.9   # деквантизация 4-bit весов на каждом шаге (замер с Unsloth, Qwen3-4B: 565 против 610 ток/с)
UNSLOTH_SPEEDUP = 1.6  # собственные ядра Unsloth для поддерживаемых архитектур
NATIVE_BONUS = 3.0  # пункта MMLU-Pro за язык задачи в списке поддерживаемых
TYPICAL = {"examples": 10_000, "tokens_per_example": 300, "epochs": 3, "seq_len": 512}


def assess_model(model: ModelInfo, hw, safety: float = 0.85, budget_hours: float | None = None) -> dict:
    """Пригодность модели для этой машины: режим (LoRA bf16 / только QLoRA / не влезает), RAM, диск, время.

    Время — на типовой объём (10 тыс. примеров × 300 токенов × 3 эпохи) из замеренных TFLOPS:
    ~4·N FLOP на токен при обучении адаптера поверх замороженной базы (прямой 2N + обратный по активациям 2N;
    градиенты весов базы не считаются); у MoE N — активные параметры, память — по всем."""
    seq = TYPICAL["seq_len"]
    out = {"id": model.id, "params_b": model.params_b, "quality": model.quality, "source": model.source,
           "mode": "none", "peak_gb": None, "hours_typical": None, "ram_ok": True, "disk_ok": True, "problems": []}
    if not hw.has_gpu:
        out["problems"].append("нет GPU — обучение не поддерживается")
        out["fits"] = False
        return out
    vram = getattr(hw, "usable_vram_gb", hw.vram_gb)  # без памяти, занятой другими программами
    modes = (("4bit", "qlora"),) if model.prequantized else (("none", "lora"), ("4bit", "qlora"))
    for quant, mode in modes:
        plan = plan_memory(model, seq, vram, safety, quant)
        if plan["fits"]:
            out.update(mode=mode, peak_gb=plan["estimate"]["total_gb"], micro_batch=plan["micro_batch"],
                       gradient_checkpointing=plan["gradient_checkpointing"])
            break
    if out["mode"] == "none":
        out["problems"].append(f"не помещается в {vram:.1f} ГБ свободной VRAM даже в 4 битах")
    # веса на диске: bf16 (загружаются и для QLoRA — квантуются при загрузке) или уже 4-битные
    weights_gb = model.params_b * (0.55 if model.prequantized else 2)
    out["download_gb"] = round(weights_gb, 1)
    ram_need = weights_gb * 0.5 + 2  # загрузка по шардам: пик ~половина весов + интерпретатор
    if hw.ram_gb and ram_need > hw.ram_gb:
        out["ram_ok"] = False
        out["problems"].append(f"нужно ~{ram_need:.0f} ГБ RAM для загрузки, есть {hw.ram_gb:.0f}")
    if hw.disk_free_gb and weights_gb * 1.2 > hw.disk_free_gb:
        out["disk_ok"] = False
        out["problems"].append(f"нужно ~{weights_gb * 1.2:.0f} ГБ на диске, свободно {hw.disk_free_gb:.0f}")
    if hw.measured_tflops and out["mode"] != "none":
        tps = hw.measured_tflops * 1e12 * MFU / (4 * (model.active_params_b or model.params_b) * 1e9)
        if out["mode"] == "qlora":
            tps *= QLORA_SLOWDOWN
        if model.unsloth:
            from factory.hparams.guard import unsloth_available
            tps *= UNSLOTH_SPEEDUP if unsloth_available() else 1.0
        tokens = TYPICAL["examples"] * TYPICAL["tokens_per_example"] * TYPICAL["epochs"]
        out["tokens_per_sec_est"] = round(tps)
        out["hours_typical"] = round(tokens / tps / 3600, 2)
        if budget_hours and out["hours_typical"] > budget_hours:
            out["problems"].append(f"типовой объём займёт ~{out['hours_typical']:.1f} ч при бюджете {budget_hours} ч")
    out["fits"] = out["mode"] != "none" and out["ram_ok"] and out["disk_ok"]
    return out


def rank_models(models: list[ModelInfo], hw, safety: float = 0.85, budget_hours: float | None = None,
                language: str | None = None) -> list[dict]:
    """Оценка всех моделей + рекомендация: лучшая по качеству (quality) из помещающихся в память этой GPU —
    в LoRA bf16 или в QLoRA (квантование при загрузке либо готовый 4-битный чекпоинт). Память считается честно:
    с KV-кэшем генерации на оценке (estimate_memory) и без VRAM, занятой другими программами. При равном качестве —
    LoRA bf16 (точнее и быстрее), затем меньший объём скачивания (готовая 4-битная копия). Размер сам по себе не
    довод: новое поколение того же размера заметно сильнее старого. Время ограничивает выбор, только если
    пользователь задал бюджет GPU-часов. Явная поддержка языка задачи — бонус NATIVE_BONUS к качеству, а не фильтр:
    у многих сильных моделей карточка не перечисляет языки вовсе. Ничего не помещается — самая лёгкая."""
    rows = [assess_model(m, hw, safety, budget_hours) for m in models]
    native = {m.id for m in models if language and language in m.languages}
    for r in rows:
        r["native_language"] = r["id"] in native
    in_time = lambda r: not budget_hours or r["hours_typical"] is None or r["hours_typical"] <= budget_hours  # noqa: E731
    pool = [r for r in rows if r["fits"] and in_time(r)]
    if pool:
        max(pool, key=lambda r: (r["quality"] + NATIVE_BONUS * r["native_language"], r["mode"] == "lora",
                                 -r["download_gb"], r["params_b"]))["recommended"] = True
    else:
        fitting = [r for r in rows if r["fits"]]
        if fitting:
            min(fitting, key=lambda r: r["params_b"])["recommended"] = True
    for r in rows:
        r.setdefault("recommended", False)
    order = {"lora": 0, "qlora": 1, "none": 2}
    return sorted(rows, key=lambda r: (not r["recommended"], not r["fits"], -r["quality"], order[r["mode"]],
                                       -r["params_b"]))
