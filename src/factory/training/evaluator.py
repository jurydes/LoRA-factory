"""Оценка модели генерацией на подвыборке (val во время обучения, test при окончательной остановке)."""
from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from factory.hparams.resources import EVAL_GEN_BATCH, KV_OVERHEAD
from factory.training import metrics as metrics_mod
from factory.training.formatting import PromptFn, extract_answer

logger = logging.getLogger(__name__)


MAX_GEN_BATCH = 16  # замер (Qwen3-1.7B, RTX 3060): батч 8 → 56 ток/с, 16 → 60 ток/с — дальше выигрыша нет
LOW_VRAM_SHARE = 0.1  # свободной видеопамяти меньше — батч уменьшается


GB = 1024 ** 3
VRAM_MARGIN_GB = 0.6  # CUDA-контекст, рабочая память cuBLAS, ядра Triton — вне аллокатора PyTorch
_cap_bytes: int | None = None


def cap_vram() -> float | None:
    """Предел памяти PyTorch для процесса: свободная сейчас видеопамять минус запас. На Windows перебор VRAM не даёт
    ошибки: драйвер выносит часть данных в оперативную память, и обучение молча замедляется в десятки раз. С пределом
    перебор — обычная нехватка памяти (OutOfMemoryError), её разбирают генерация (меньше батч) и супервизор
    (лестница: micro-batch → checkpointing → QLoRA). Память, занятая другими программами, в предел не входит.
    Возвращает предел в ГБ (None — без GPU)."""
    global _cap_bytes
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    _cap_bytes = max(int(free - VRAM_MARGIN_GB * GB), GB)
    torch.cuda.set_per_process_memory_fraction(min(1.0, _cap_bytes / total))
    return _cap_bytes / GB


def _available_bytes() -> int:
    """Сколько ещё можно выделить: свободное на карте + кэш аллокатора PyTorch, но не сверх предела процесса."""
    free, _ = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    free += torch.cuda.memory_reserved() - allocated
    return min(free, _cap_bytes - allocated) if _cap_bytes else free


def memory_note() -> str:
    """Занято тензорами / резерв аллокатора / пик — для журнала."""
    if not torch.cuda.is_available():
        return "без GPU"
    return (f"занято {torch.cuda.memory_allocated() / GB:.2f} ГБ, резерв {torch.cuda.memory_reserved() / GB:.2f} ГБ, "
            f"пик {torch.cuda.max_memory_allocated() / GB:.2f} ГБ")


def _text_config(model):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def auto_batch_size(model, max_tokens: int, cap: int = MAX_GEN_BATCH) -> int:
    """Батч генерации по свободной видеопамяти: KV-кэш (K и V по слоям, bf16) на последовательность длиной
    max_tokens с запасом KV_OVERHEAD, бюджет — 60% свободного. От микробатча обучения не зависит: обучению память
    нужна на градиенты и состояния оптимизатора, генерации — нет. Перебор памяти на Windows не даёт ошибки:
    драйвер выносит часть данных в оперативную память, и генерация замедляется в десятки раз."""
    if not torch.cuda.is_available():
        return 8
    cfg = _text_config(model)
    heads = getattr(cfg, "num_attention_heads", 32)
    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    head_dim = getattr(cfg, "head_dim", None) or getattr(cfg, "hidden_size", 4096) // heads
    types = getattr(cfg, "layer_types", None)  # гибриды (Qwen3.5): у слоёв линейного внимания KV-кэша нет
    layers = sum(1 for t in types if "linear" not in t) if types else getattr(cfg, "num_hidden_layers", 32)
    per_token = 2 * layers * kv_heads * head_dim * 2 * KV_OVERHEAD
    return max(1, min(cap, int(0.6 * _available_bytes() / (per_token * max(max_tokens, 1)))))


def _vram_low() -> bool:
    if not torch.cuda.is_available():
        return False
    return _available_bytes() < LOW_VRAM_SHARE * torch.cuda.mem_get_info()[1]


_LINE_TOKENS: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _line_token_classes(tokenizer, vocab: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """По словарю: в токене есть перевод строки; токен несёт текст; в токене есть текст до перевода строки."""
    if id(tokenizer) not in _LINE_TOKENS:
        texts = tokenizer.batch_decode([[i] for i in range(len(tokenizer))])
        pad = [False] * max(0, vocab - len(texts))
        _LINE_TOKENS[id(tokenizer)] = (
            torch.tensor(["\n" in s for s in texts] + pad), torch.tensor([bool(s.strip()) for s in texts] + pad),
            torch.tensor([bool(s.split("\n", 1)[0].strip()) for s in texts] + pad))
    return _LINE_TOKENS[id(tokenizer)]


class FirstLineStop:
    """Ответ — первая непустая строка (extract_answer): последовательность готова, как только после текста пошёл
    перевод строки. Без этого генерация шла до max_new_tokens, а всё после первой строки выбрасывалось.
    Критерий остановки transformers (StoppingCriteria): по одному признаку «готово» на последовательность."""

    def __init__(self, tokenizer, vocab: int, batch: int, device):
        nl, text, before = _line_token_classes(tokenizer, vocab)
        self.nl, self.text, self.before = nl.to(device), text.to(device), before.to(device)
        self.seen = torch.zeros(batch, dtype=torch.bool, device=device)

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        last = input_ids[:, -1]
        done = self.nl[last] & (self.seen | self.before[last])
        self.seen |= self.text[last]
        return done


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int | None = None,
             max_prompt_tokens: int | None = None, unsloth: bool = False, repetition_penalty: float = 1.0,
             on_batch: Callable[[int, int], None] | None = None) -> list[str]:
    """Пакетная жадная генерация с левым паддингом; возвращает первую строку ответа.

    Промпты идут от длинных к коротким: в батче меньше паддинга, а нехватка памяти обнаруживается на первом батче.
    batch_size=None — по свободной памяти GPU; при нехватке памяти батч уменьшается вдвое. Шаг генерации на Windows
    упирается в CPU (накладные расходы на запуск ядер), поэтому большой батч почти не замедляет шаг.
    on_batch(готово, всего) — после каждого батча: ход генерации и признак жизни процесса."""
    from transformers import StoppingCriteriaList  # здесь, а не в модуле: Unsloth импортируется раньше transformers

    was_training = model.training
    if unsloth:  # Unsloth: режим генерации включается явно и выключается после
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    model.eval()
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    out = [""] * len(prompts)
    # промпт из шаблона чата уже содержит BOS — второй не добавляем
    special = not (tokenizer.bos_token and prompts and prompts[0].startswith(tokenizer.bos_token))
    lengths = [len(ids) for ids in tokenizer(prompts, add_special_tokens=special)["input_ids"]] if prompts else []
    if max_prompt_tokens:
        lengths = [min(n, max_prompt_tokens) for n in lengths]
    order = sorted(range(len(prompts)), key=lambda i: -lengths[i])
    size = batch_size or auto_batch_size(model, max(lengths, default=1) + max_new_tokens)
    if was_training:  # оценка внутри обучения: память заложена в план раунда только на EVAL_GEN_BATCH последовательностей
        size = min(size, EVAL_GEN_BATCH)
    vocab = max(getattr(_text_config(model), "vocab_size", 0) or 0, len(tokenizer))
    try:
        i = 0
        while i < len(order):
            idx = order[i:i + size]
            batch = [prompts[j] for j in idx]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=max_prompt_tokens is not None,
                            max_length=max_prompt_tokens, add_special_tokens=special).to(model.device)
            try:
                gen = model.generate(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=max_new_tokens,
                    do_sample=False, repetition_penalty=repetition_penalty, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, use_cache=True,
                    stopping_criteria=StoppingCriteriaList([FirstLineStop(tokenizer, vocab, len(batch), model.device)]))
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                del enc
                torch.cuda.empty_cache()
                size = max(1, size // 2)
                logger.warning("Генерации не хватило памяти GPU — батч уменьшен до %d", size)
                continue
            texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, t in zip(idx, texts):
                out[j] = extract_answer(t)
            i += len(idx)
            del gen, enc
            if size > 1 and _vram_low():  # на грани: следующий батч мог бы уйти в оперативную память
                size //= 2
                logger.warning("Видеопамять почти занята — батч генерации уменьшен до %d", size)
            if on_batch:
                on_batch(i, len(order))
    finally:
        tokenizer.padding_side = side
        if unsloth:
            release_unsloth_buffers(model)
        if unsloth and was_training:
            FastLanguageModel.for_training(model)
        if was_training:
            model.train()
        # KV-кэш генерации (гигабайты на длинных промптах) остаётся в резерве аллокатора PyTorch; обучению после
        # оценки нужны блоки других размеров — резерв растёт сверх VRAM, и Windows выносит часть в оперативную
        # память без ошибки (шаг замедляется в десятки раз). Ни Unsloth (for_training лишь меняет флаги), ни
        # Trainer резерв не отдают.
        if torch.cuda.is_available():
            gc.collect()  # тензоры в циклических ссылках (объекты генерации) живут до сборки мусора
            torch.cuda.empty_cache()
    return out


# Буферы быстрой генерации Unsloth на модулях внимания (models/llama.py, LlamaAttention_fast_forward_inference):
# KV-кэш на батч × (промпт + запас) в каждом слое. Unsloth удаляет их лениво — в первом прямом проходе обучения,
# слой за слоем, так что первый шаг после оценки идёт с гигабайтами чужих буферов (Qwen3-4B, батч 16, промпты
# ~900 токенов — ~3 ГБ): перебор VRAM.
UNSLOTH_BUFFERS = ("paged_attention_K", "paged_attention_V", "paged_attention", "temp_QA", "temp_KV", "RH_Q",
                   "attention", "temp_O")


def release_unsloth_buffers(model) -> None:
    for module in model.modules():
        if hasattr(module, "paged_attention"):
            for name in UNSLOTH_BUFFERS:
                if isinstance(getattr(module, name, None), torch.Tensor):
                    delattr(module, name)



@dataclass
class TaskEvaluator:
    fmt: str  # формат ответа задачи: label | text | entities | span
    metric_names: list[str]
    prompt: PromptFn
    inputs: list[str]
    references: list[str]
    labels: list[str] | None
    max_new_tokens: int
    batch_size: int | None = None  # None — по свободной памяти GPU (auto_batch_size)
    on_progress: Callable[[int, int], None] | None = None  # (готово, всего) после каждого батча генерации
    raw_inputs: list[str] | None = None  # входы до обрезки: для простых baseline'ов и примеров в отчёте
    alternatives: list[list[str]] | None = None  # все эталонные ответы (QA)
    unsloth: bool = False

    @property
    def original_inputs(self) -> list[str]:
        return self.raw_inputs if self.raw_inputs is not None else self.inputs

    @property
    def repetition_penalty(self) -> float:
        """Штраф за повтор действует и на токены промпта: метки перечислены в инструкции, сущности и ответ QA
        копируются из входа — для них штраф мешает. Нужен только свободному тексту против зацикливания."""
        return 1.1 if self.fmt == "text" else 1.0

    def predict(self, model, tokenizer) -> list[str]:
        return generate(model, tokenizer, [self.prompt(i) for i in self.inputs], self.max_new_tokens, self.batch_size,
                        unsloth=self.unsloth, repetition_penalty=self.repetition_penalty, on_batch=self.on_progress)

    def score_predictions(self, preds: list[str], cheap_only: bool = False) -> dict:
        return metrics_mod.compute(self.fmt, self.metric_names, preds, self.references, self.labels, cheap_only,
                                   self.alternatives)

    def run(self, model, tokenizer, cheap_only: bool = True) -> dict:
        started = time.monotonic()
        preds = self.predict(model, tokenizer)
        res = self.score_predictions(preds, cheap_only)
        res["predictions"] = preds
        res["pred_label_share"] = metrics_mod.prediction_shares(res["normalized_predictions"]) if self.fmt == "label" else None
        res["elapsed_s"] = time.monotonic() - started
        return res

    def latency_ms(self, model, tokenizer, n: int = 10) -> float:
        """Средняя задержка одного запроса (батч 1) — для ограничения max_latency_ms."""
        prompts = [self.prompt(i) for i in self.inputs[:n]]
        if not prompts:
            return 0.0
        started = time.monotonic()
        for p in prompts:
            generate(model, tokenizer, [p], self.max_new_tokens, batch_size=1, unsloth=self.unsloth,
                     repetition_penalty=self.repetition_penalty)
        return (time.monotonic() - started) * 1000 / len(prompts)
