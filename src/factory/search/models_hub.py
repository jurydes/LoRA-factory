"""Базовые модели с HF Hub под машину пользователя: свежие официальные instruct-модели, которые помещаются в его GPU.

Реестр (registry/models.yaml) — проверенный вручную минимум. Каталог дополняет его тем, что есть на HF сейчас,
и подбирается под объём видеопамяти этой машины (12 ГБ и 80 ГБ дают разные списки):
1) популярные и набирающие популярность модели — текстовые и мультимодальные (обучается языковая часть) —
   и готовые 4-битные копии (bitsandbytes): QLoRA без квантования при загрузке, скачивание вчетверо меньше;
2) фильтр: официальный релиз (не перезалив, не дообучение третьих лиц) или bnb-4bit копия официального релиза,
   есть чат-шаблон, без собственного кода (trust_remote_code), по размеру хотя бы в 4 битах влезает в эту GPU;
   GGUF, AWQ, GPTQ, FP8 не обучаются здесь (нет бэкенда) и отсекаются;
3) архитектура — в подпроцессе на meta-устройстве (без весов): размеры, имена линейных слоёв для LoRA, активные
   параметры MoE; модели, которых не знает установленный transformers, отпадают здесь;
4) качество — LLM читает карточку модели и выписывает MMLU-Pro и список языков; число принимается, только если оно
   есть в тексте карточки.
Кэш — DATA_DIR/cache/models: каталог обновляется раз в неделю или при смене GPU, разбор карточки — при смене
ревизии репозитория."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

from factory.core import llm
from factory.core.config import AppConfig, get_config
from factory.core.registry import ModelInfo, catalog_path
from factory.core.storage import read_json, write_json

logger = logging.getLogger(__name__)

PIPELINES = ("text-generation", "image-text-to-text", "any-to-any")
SORTS = ("downloads", "trending_score")  # проверенные временем и свежие релизы
PER_QUERY = 200
EXPAND = ["safetensors", "config", "cardData", "gated", "downloads", "createdAt", "tags", "sha"]
QUANT_TAGS = {"gguf", "awq", "gptq", "mlx", "exl2", "fp8", "8-bit", "nvfp4", "onnx", "openvino", "compressed-tensors"}
BNB_TAGS = {"bitsandbytes", "4-bit"}  # готовые 4-битные веса bitsandbytes — обучаются как QLoRA
# не модели общего назначения или не исходные веса
# (think — модели только с рассуждениями: ответ длинной цепочкой, не первой строкой; assistant — черновые модели
# спекулятивного декодирования; qat — копии под квантизацию)
SKIP_NAME = re.compile(r"coder|math|guard|reward|embed|rerank|ocr|tts|asr|speech|audio|omni|prover|safety|"
                       r"uncensored|abliterated|heretic|draft|eagle|assistant|think|r1-distill|qat|"
                       r"gguf|awq|gptq|fp8|fp4|int4|int8|mlx|-base$", re.I)
MIN_PARAMS_B = 0.3
MIN_DOWNLOADS = 2_000
MAX_INSPECT = 200  # разбор архитектуры: ~1 с на модель
MAX_CARDS = 40  # карточек на одно обновление читает LLM (прочитанные берутся из кэша)
CATALOG_TTL_S = 7 * 24 * 3600
CARD_PARSER = 3  # версия разбора карточек: при изменении разбора прочитанные карточки читаются заново
# архитектуры, которые грузит FastLanguageModel (остальные учатся через transformers + PEFT)
UNSLOTH_TYPES = {"llama", "mistral", "qwen2", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe", "gemma", "gemma2",
                 "gemma3", "gemma3_text", "gemma4", "phi3", "granite", "cohere"}
OPEN_LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "gemma", "llama3", "llama3.1", "llama3.2", "llama3.3",
                 "llama4", "cc-by-4.0", "openrail", "qwen"}
DEFAULT_THINKING = r"thinking mode by default|thinks by default|reasoning (mode )?by default"
THINKING_GAP = 10.0  # MMLU-Pro в режиме рассуждений выше, чем без него (Qwen3.5-2B: 66.5 против 55.3)


def _models_dir(data_dir: Path) -> Path:
    return data_dir / "cache" / "models"


# ---------------------------------------------------------------- 1–2. кандидаты с HF Hub

def _params_b(m) -> float | None:
    total = (m.safetensors or {}).get("total") if isinstance(m.safetensors, dict) else getattr(m.safetensors, "total", None)
    return total / 1e9 if total else None


def _has_chat_template(config: dict) -> bool:
    return bool((config.get("tokenizer_config") or {}).get("chat_template") or config.get("chat_template_jinja"))


def _official(model_id: str, card: dict) -> bool:
    """Релиз автора модели: базовая модель не указана или того же автора (instruct к своей base)."""
    base = card.get("base_model")
    bases = [base] if isinstance(base, str) else list(base or [])
    author = model_id.split("/")[0]
    return all(str(b).split("/")[0] == author for b in bases)


def _card_languages(card: dict) -> list[str]:
    lang = card.get("language")
    return [lang] if isinstance(lang, str) else [str(x) for x in (lang or [])]


def _bnb4(m) -> bool:
    tags = set(m.tags or [])
    return BNB_TAGS <= tags and not QUANT_TAGS & tags


def candidate(m, vram_gb: float) -> dict | None:
    """Запись списка HF → кандидат или None (не подходит по признакам из метаданных). У bnb-4bit копии base —
    исходная модель: копия остаётся, только если та — официальный релиз (проверяет hub_candidates)."""
    card = m.card_data.to_dict() if m.card_data else {}
    config = m.config or {}
    params = _params_b(m)
    if params is None or params < MIN_PARAMS_B or params * 0.55 + 1.5 > vram_gb:  # не влезет даже в 4 битах
        return None
    if (m.downloads or 0) < MIN_DOWNLOADS or SKIP_NAME.search(m.id.split("/")[-1]):
        return None
    if QUANT_TAGS & set(m.tags or []) or "auto_map" in config or not _has_chat_template(config):
        return None
    base = card.get("base_model")
    prequantized = _bnb4(m)
    if prequantized:
        if not isinstance(base, str) and len(base or []) != 1:
            return None
        base = base if isinstance(base, str) else base[0]
    elif not _official(m.id, card):
        return None
    elif BNB_TAGS & set(m.tags or []):  # 8-bit и прочие варианты bitsandbytes
        return None
    license_ = str(card.get("license") or "")
    return {"id": m.id, "params_b": round(params, 2), "downloads": m.downloads or 0, "sha": m.sha,
            "created": str(m.created_at or "")[:10], "license": license_, "languages": _card_languages(card),
            "gated": bool(m.gated), "model_type": config.get("model_type"), "prequantized": prequantized,
            "base": base if prequantized else None}


def hub_candidates(vram_gb: float, api=None, known: set[str] = frozenset()) -> list[dict]:
    """Кандидаты под эту GPU. known — id, которым можно доверять как исходным моделям квантованных копий (реестр);
    кроме них — официальные релизы из тех же списков, независимо от размера (bf16 не влезет, а 4-битная копия —
    да). Из нескольких копий одной модели остаётся самая скачиваемая."""
    from huggingface_hub import HfApi
    api = api or HfApi(token=_hf_token())
    found: dict[str, dict] = {}
    official = set(known)
    queries = [dict(pipeline_tag=p, sort=s) for p in PIPELINES for s in SORTS]
    queries += [dict(search="bnb-4bit", sort=s) for s in SORTS]
    for q in queries:
        try:
            for m in api.list_models(limit=PER_QUERY, expand=EXPAND, **q):
                card = m.card_data.to_dict() if m.card_data else {}
                if not _bnb4(m) and _official(m.id, card):
                    official.add(m.id)
                c = candidate(m, vram_gb)
                if c:
                    found.setdefault(c["id"], c)
        except Exception as e:
            logger.warning("HF Hub: список %s не получен: %s", q, e)
    copies: dict[str, dict] = {}
    for c in sorted(found.values(), key=lambda c: -c["downloads"]):
        if c["prequantized"]:
            if c["base"] in official:
                copies.setdefault(c["base"], c)
    out = [c for c in found.values() if not c["prequantized"]] + list(copies.values())
    return sorted(out, key=lambda c: -c["downloads"])


# ---------------------------------------------------------------- 3. архитектура (в подпроцессе)

def inspect_config(raw: dict) -> dict | None:
    """config.json → размеры и имена линейных слоёв; модель строится на meta-устройстве, без весов.
    None — архитектуру не знает установленный transformers или это не языковая модель."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    raw = {k: v for k, v in raw.items() if k not in ("auto_map", "quantization_config")}  # архитектура — без весов
    try:
        cfg = AutoConfig.for_model(**raw)
    except (ValueError, KeyError, TypeError):
        return None
    text = cfg.get_text_config()
    model = None
    with init_empty_weights():
        for c in (text, cfg):
            try:
                model = AutoModelForCausalLM.from_config(c)
                break
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
    if model is None:
        return None
    named = {n: p for n, p in model.named_parameters() if not re.search(r"vision|visual|audio|image", n)}
    total = sum(p.numel() for p in named.values())
    head = model.get_output_embeddings()
    tied = getattr(text, "tie_word_embeddings", getattr(cfg, "tie_word_embeddings", False))
    if tied and head is not None and head.weight is not model.get_input_embeddings().weight:
        total -= head.weight.numel()  # на meta-устройстве связанные веса не связываются — не считать дважды
    experts = sum(p.numel() for n, p in named.items() if ".experts." in n)
    n_experts = next((getattr(text, k) for k in ("num_experts", "num_local_experts", "n_routed_experts")
                      if getattr(text, k, None)), None)
    top_k = getattr(text, "num_experts_per_tok", None)
    active = total - experts + experts * top_k / n_experts if experts and n_experts and top_k else None
    attention, all_linear = [], []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or re.search(r"vision|visual|audio|image|lm_head", name):
            continue
        leaf = name.split(".")[-1]
        if leaf in ("gate", "router") or "expert_gate" in leaf or "per_layer" in leaf or mod.out_features < text.hidden_size // 16:
            continue  # роутер MoE, гейты голов, проекции per-layer эмбеддингов Gemma
        if leaf not in all_linear:
            all_linear.append(leaf)
        if re.search(r"attn|attention|mixer", name) and leaf not in attention:
            attention.append(leaf)
    if not all_linear:
        return None
    composite = any(getattr(cfg, k, None) is not None for k in ("vision_config", "audio_config"))
    heads = getattr(text, "num_attention_heads", None) or 1
    kv_heads = getattr(text, "num_key_value_heads", None) or heads
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // heads
    types = getattr(text, "layer_types", None)  # у слоёв линейного внимания (Qwen3.5) KV-кэша нет
    kv_layers = sum(1 for t in types if "linear" not in str(t)) if types else text.num_hidden_layers
    return {"params_b": round(total / 1e9, 3), "active_params_b": round(active / 1e9, 3) if active else None,
            "kv_bytes_per_token": 2 * kv_layers * kv_heads * head_dim * 2,
            "hidden_size": text.hidden_size, "num_layers": text.num_hidden_layers,
            "vocab_size": text.vocab_size, "max_context": getattr(text, "max_position_embeddings", None) or 4096,
            "model_type": cfg.model_type, "text_only": composite,
            "lora_modules": {"attention": attention or all_linear, "all_linear": all_linear}}


def _inspect_many(ids: list[str]) -> dict[str, dict | None]:
    from huggingface_hub import hf_hub_download
    out = {}
    for model_id in ids:
        try:
            raw = json.loads(Path(hf_hub_download(model_id, "config.json")).read_text(encoding="utf-8"))
            out[model_id] = inspect_config(raw)
        except Exception as e:
            logger.info("Архитектура %s не разобрана: %s", model_id, e)
            out[model_id] = None
    return out


def _hf_token() -> str | None:
    try:
        from settings import load_global_config
        return load_global_config().hf_token or None
    except Exception:
        return None


async def inspect_architectures(ids: list[str], timeout_s: float = 900) -> dict[str, dict | None]:
    """Разбор в отдельном процессе: импорт transformers и построение моделей держат GIL и память сервера."""
    from factory.core.hardware import _child_env

    env = _child_env()
    token = _hf_token()
    if token:  # gated-модели (Llama, Gemma): config.json доступен только с токеном и принятой лицензией
        env["HF_TOKEN"] = token

    def run() -> dict:
        out = subprocess.run([sys.executable, "-m", "factory.search.models_hub"], input=json.dumps(ids),
                             capture_output=True, encoding="utf-8", errors="replace", timeout=timeout_s,
                             env=env, cwd=str(Path(__file__).parents[2]))
        lines = out.stdout.strip().splitlines()
        if out.returncode or not lines:
            raise RuntimeError(f"код {out.returncode}: {out.stderr[-500:]}")
        return json.loads(lines[-1])
    return await asyncio.to_thread(run)


# ---------------------------------------------------------------- 4. качество из карточки

class CardFacts(BaseModel):
    mmlu_pro: float | None = Field(None, description=(
        "MMLU-Pro именно этой модели (этого размера и этой версии, не соседних и не конкурентов) в процентах, "
        "как в карточке. Есть оба режима — бери без рассуждений (non-thinking / instruct). Числа предобученной "
        "(base / pt) версии не бери. Нет в карточке — null."))
    mmlu_pro_thinking: bool = Field(False, description=(
        "число получено в режиме рассуждений (thinking/CoT): так подписана таблица, или модель по умолчанию "
        "рассуждает и про режим без рассуждений для этого числа не сказано"))
    languages: list[str] = Field(default_factory=list, description=(
        "ISO 639-1 коды языков, которые карточка явно называет поддерживаемыми; не сказано — пустой список"))
    instruct: bool = Field(True, description="модель дообучена для диалога/инструкций (не базовая)")
    reasoning_only: bool = Field(False, description="модель всегда рассуждает перед ответом, режима без рассуждений нет")


CARD_INSTRUCTIONS = """Ты читаешь карточку языковой модели с Hugging Face и выписываешь факты строго из её текста.
Не подставляй числа по памяти и не путай столбцы таблиц: в одной таблице часто несколько моделей и размеров."""


def _html_rows(text: str) -> str:
    """HTML-таблицы (<tr><td>…) → строки markdown-таблиц: часть карточек (Qwen3.5) свёрстана в HTML."""
    def row(m: re.Match) -> str:
        cells = [re.sub(r"<[^>]+>|\s+", " ", c).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", m.group(0), re.S)]
        return "\n| " + " | ".join(cells) + " |\n" if cells else ""
    return re.sub(r"<tr[^>]*>.*?</tr>", row, text, flags=re.S | re.I)


def condense_card(text: str, limit: int = 24_000) -> str:
    """Таблицы, заголовки и строки о языках — там бенчмарки и список языков; остальное отбрасывается."""
    keep = [line for line in _html_rows(text).splitlines()
            if line.lstrip().startswith(("|", "#"))
            or re.search(r"language|multilingual|язык|thinking|reasoning|non-thinking|рассужд", line, re.I)]
    return "\n".join(keep)[:limit]


def number_in_text(value: float, text: str) -> bool:
    variants = {f"{value:g}", f"{value:.1f}", f"{value:.2f}"}
    return any(re.search(rf"(?<![\d.]){re.escape(v)}(?!\d)", text) for v in variants)


def _card_text(model_id: str) -> str:
    from huggingface_hub import ModelCard
    return ModelCard.load(model_id, token=_hf_token()).text


async def read_card(model_id: str, text: str) -> CardFacts | None:
    body = condense_card(text)
    if not body:
        return None
    facts = await llm.complete("model_scout", f"Модель: {model_id}\n\nКарточка (таблицы и строки о языках):\n{body}",
                               CardFacts, CARD_INSTRUCTIONS)
    if facts.mmlu_pro is not None and not number_in_text(facts.mmlu_pro, body):
        logger.info("%s: MMLU-Pro %s нет в тексте карточки — отброшено", model_id, facts.mmlu_pro)
        facts.mmlu_pro = None
    # LLM отмечает режим непоследовательно (одна таблица — разные ответы для соседних столбцов): модель по умолчанию
    # рассуждает, а таблицы без рассуждений в карточке нет — число получено с рассуждениями
    if re.search(DEFAULT_THINKING, body, re.I) and not re.search(r"non[- ]thinking", body, re.I):
        facts.mmlu_pro_thinking = True
    return facts


def quality_of(facts: CardFacts | None) -> tuple[float, str]:
    if not facts or facts.mmlu_pro is None:
        return 0.0, "MMLU-Pro в карточке нет"
    if facts.mmlu_pro_thinking:
        return round(facts.mmlu_pro - THINKING_GAP, 1), f"MMLU-Pro {facts.mmlu_pro:g} в режиме рассуждений, −{THINKING_GAP:g}"
    return facts.mmlu_pro, f"MMLU-Pro {facts.mmlu_pro:g} по карточке"


# ---------------------------------------------------------------- сборка каталога

def to_model_info(c: dict, arch: dict, facts: CardFacts | None) -> ModelInfo:
    quality, note = quality_of(facts)
    languages = sorted({*c["languages"], *(facts.languages if facts else [])}) or ["multi"]
    return ModelInfo(
        id=c["id"], params_b=arch["params_b"], active_params_b=arch["active_params_b"],
        hidden_size=arch["hidden_size"], num_layers=arch["num_layers"], vocab_size=arch["vocab_size"],
        max_context=arch["max_context"], kv_bytes_per_token=arch.get("kv_bytes_per_token"),
        prequantized=c.get("prequantized", False), base_model=c.get("base"), languages=languages, license=c["license"],
        commercial_ok=c["license"] in OPEN_LICENSES, gated=c["gated"], lora_modules=arch["lora_modules"],
        unsloth=arch["model_type"] in UNSLOTH_TYPES, ptuning=True, chat_template=True, quality=quality,
        quality_note=note, text_only=arch["text_only"], source="hub")


def catalog_fresh(data_dir: Path, vram_gb: float) -> bool:
    raw = read_json(catalog_path(data_dir), {}) or {}
    return raw.get("vram_gb") == vram_gb and time.time() - raw.get("updated", 0) < CATALOG_TTL_S


async def refresh(hw, cfg: AppConfig | None = None, force: bool = False) -> list[ModelInfo]:
    """Обновляет каталог моделей под эту машину; возвращает модели каталога (свежий кэш — без обращений к сети)."""
    from factory.core.registry import Registry, load_catalog
    from factory.hparams.resources import assess_model
    cfg = cfg or get_config()
    if not hw.has_gpu:
        return []
    hw = hw.model_copy(update={"vram_others_gb": 0.0})  # каталог — про карту, а не про программы, открытые сейчас
    if not force and catalog_fresh(cfg.data_dir, hw.vram_gb):
        return load_catalog(cfg.data_dir)
    registry = Registry.from_dir(cfg.registry_dir).models
    cands = (await asyncio.to_thread(hub_candidates, hw.vram_gb, None, set(registry)))[:MAX_INSPECT]
    arch = await inspect_architectures([c["id"] for c in cands])
    cards_path = _models_dir(cfg.data_dir) / "cards.json"
    cards: dict = read_json(cards_path, {}) or {}
    fitting = []
    for c in cands:
        a = arch.get(c["id"])
        if a and assess_model(to_model_info(c, a, None), hw, cfg.hparams.vram_safety)["fits"]:
            fitting.append((c, a))
    sem = asyncio.Semaphore(4)

    async def facts_for(c: dict) -> CardFacts | None:
        cached = cards.get(c["id"])
        if cached and cached.get("sha") == c["sha"] and cached.get("parser") == CARD_PARSER:
            return CardFacts.model_validate(cached["facts"]) if cached.get("facts") else None
        async with sem:
            try:
                facts = await read_card(c["id"], await asyncio.to_thread(_card_text, c["id"]))
            except Exception as e:
                logger.warning("Карточка %s не прочитана: %s", c["id"], e)
                return None
        cards[c["id"]] = {"sha": c["sha"], "parser": CARD_PARSER, "facts": facts.model_dump() if facts else None}
        return facts

    originals = [(c, a) for c, a in fitting if not c.get("prequantized")]
    copies = [(c, a) for c, a in fitting if c.get("prequantized")]
    with llm.llm_log_to(_models_dir(cfg.data_dir) / "llm_calls.jsonl"):
        facts = await asyncio.gather(*(facts_for(c) for c, _ in originals[:MAX_CARDS]))
    facts += [None] * (len(originals) - len(facts))
    rejected = {c["id"] for (c, _), f in zip(originals, facts) if f and (not f.instruct or f.reasoning_only)}
    models = [to_model_info(c, a, f) for (c, a), f in zip(originals, facts) if c["id"] not in rejected]
    # квантованная копия: качество, языки и лицензия — исходной модели (реестр точнее каталога); карточку копии
    # не читаем — обычно это перепечатка карточки исходной
    bases = {m.id: m for m in models} | registry
    for c, a in copies:
        base = bases.get(c["base"])
        if c["base"] in rejected or base is None:
            continue
        m = to_model_info(c, a, None)
        models.append(m.model_copy(update={
            "quality": base.quality, "quality_note": f"как у исходной {base.id}: {base.quality_note or 'реестр'}",
            "languages": base.languages, "license": base.license or m.license, "commercial_ok": base.commercial_ok,
            "gated": m.gated, "text_only": a["text_only"]}))
    write_json(cards_path, cards)
    write_json(catalog_path(cfg.data_dir), {"updated": time.time(), "vram_gb": hw.vram_gb,
                                            "models": [m.model_dump(mode="json") for m in models]})
    logger.info("Каталог моделей HF: %d кандидатов, %d помещаются в %.0f ГБ, с оценкой качества %d",
                len(cands), len(models), hw.vram_gb, sum(1 for m in models if m.quality))
    return models


if __name__ == "__main__":  # подпроцесс разбора архитектур: JSON-список id в stdin → JSON {id: архитектура} в stdout
    print(json.dumps(_inspect_many(json.loads(sys.stdin.read()))))
