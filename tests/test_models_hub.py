"""Подбор моделей с HF Hub под машину: фильтры метаданных, разбор архитектуры, качество из карточки, каталог."""
import asyncio
from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse, ToolCallPart

from factory.core.hardware import Hardware
from factory.search import models_hub as mh


def hub_model(model_id, params_b=4.0, downloads=10_000, tags=(), base=None, chat=True, auto_map=False, language=None,
              license_="apache-2.0"):
    card = SimpleNamespace(to_dict=lambda: {"base_model": base, "language": language, "license": license_})
    config = {"model_type": "qwen3", "tokenizer_config": {"chat_template": "{{x}}"} if chat else {}}
    if auto_map:
        config["auto_map"] = {"AutoModelForCausalLM": "x.y"}
    return SimpleNamespace(id=model_id, safetensors={"total": params_b * 1e9}, card_data=card, config=config,
                           downloads=downloads, tags=list(tags), gated=False, sha="abc", created_at="2026-01-01")


def test_candidate_filters():
    assert mh.candidate(hub_model("Org/Good-4B-Instruct"), 12)["params_b"] == 4.0
    assert mh.candidate(hub_model("Org/Big-70B", params_b=70), 12) is None  # не влезет даже в 4 битах
    assert mh.candidate(hub_model("Org/Big-70B", params_b=70), 80)
    assert mh.candidate(hub_model("Org/M-4B-GGUF", tags=["gguf"]), 12) is None
    assert mh.candidate(hub_model("Someone/Finetune-4B", base="Org/Good-4B"), 12) is None  # дообучение третьих лиц
    assert mh.candidate(hub_model("Org/Good-4B-Instruct", base="Org/Good-4B-Base"), 12)  # instruct к своей base
    assert mh.candidate(hub_model("Org/Base-4B", chat=False), 12) is None
    assert mh.candidate(hub_model("Org/Custom-4B", auto_map=True), 12) is None  # trust_remote_code
    assert mh.candidate(hub_model("Org/Model-4B-Thinking"), 12) is None
    assert mh.candidate(hub_model("Org/Coder-4B"), 12) is None
    assert mh.candidate(hub_model("Org/Rare-4B", downloads=10), 12) is None


def qwen3_config(**extra):
    return {"model_type": "qwen3", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 1000,
            "max_position_embeddings": 4096, "tie_word_embeddings": True, **extra}


def test_inspect_config_dense_tied_and_moe():
    dense = mh.inspect_config(qwen3_config())
    assert dense["lora_modules"]["all_linear"] == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert dense["lora_modules"]["attention"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert dense["active_params_b"] is None and not dense["text_only"] and dense["max_context"] == 4096
    untied = mh.inspect_config(qwen3_config(tie_word_embeddings=False))
    assert untied["params_b"] >= dense["params_b"]  # связанные эмбеддинги не считаются дважды
    moe = mh.inspect_config(qwen3_config(model_type="qwen3_moe", hidden_size=1024, num_attention_heads=8, head_dim=128,
                                         vocab_size=32000, num_experts=16, num_experts_per_tok=2,
                                         moe_intermediate_size=1024, decoder_sparse_step=1))
    assert moe["active_params_b"] is not None and moe["active_params_b"] < moe["params_b"]
    assert "gate" not in moe["lora_modules"]["all_linear"]  # роутер MoE — не цель LoRA
    assert mh.inspect_config({"model_type": "no_such_arch"}) is None


def test_html_tables_are_kept():
    card = ('<table><tr><td>MMLU-Pro</td><td style="x">82.5</td>\n<td>79.1</td></tr></table>\n'
            'Qwen3.5 operates in thinking mode by default.')
    body = mh.condense_card(card)
    assert "| MMLU-Pro | 82.5 | 79.1 |" in body and "thinking mode by default" in body


def test_number_must_be_in_card():
    assert mh.number_in_text(55.3, "| MMLU-Pro | 69.6 | 55.3 |")
    assert not mh.number_in_text(55.3, "| MMLU-Pro | 155.31 |")
    assert mh.quality_of(mh.CardFacts(mmlu_pro=79.1, mmlu_pro_thinking=True))[0] == 69.1
    assert mh.quality_of(None)[0] == 0.0


def card_llm(answers: dict):
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    def fn(messages, info: AgentInfo):
        prompt = next(p.content for m in messages for p in getattr(m, "parts", []) if getattr(p, "part_kind", "") == "user-prompt")
        model_id = prompt.splitlines()[0].removeprefix("Модель: ")
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=answers[model_id])])
    return FunctionModel(fn)


def test_refresh_builds_catalog_for_this_gpu(cfg, monkeypatch):
    from factory.core import llm
    from factory.core.registry import Registry
    from factory.hparams.resources import rank_models
    arch = {"params_b": 4.0, "active_params_b": None, "hidden_size": 2560, "num_layers": 36, "vocab_size": 151936,
            "max_context": 32768, "model_type": "qwen3", "text_only": False,
            "lora_modules": {"attention": ["q_proj"], "all_linear": ["q_proj", "down_proj"]}}
    cands = [{"id": f"Org/{n}", "params_b": 4.0, "downloads": 1000, "sha": "s1", "created": "2026-01-01",
              "license": "apache-2.0", "languages": [], "gated": False, "model_type": "qwen3"}
             for n in ("Strong-4B", "Weak-4B", "Halluc-4B", "Base-4B", "Huge-4B")]
    calls = []
    monkeypatch.setattr(mh, "hub_candidates", lambda vram, api=None, known=frozenset(): cands)

    async def inspect(ids):
        return {i: (None if i == "Org/Huge-4B" else arch) for i in ids}
    monkeypatch.setattr(mh, "inspect_architectures", inspect)
    monkeypatch.setattr(mh, "_card_text", lambda i: calls.append(i) or "| Model | MMLU-Pro |\n| x | 70.5 | 41.0 |\nLanguages: ru, en")
    answers = {"Org/Strong-4B": {"mmlu_pro": 70.5, "languages": ["ru", "en"]},
               "Org/Weak-4B": {"mmlu_pro": 41.0, "languages": ["en"]},
               "Org/Halluc-4B": {"mmlu_pro": 88.8},  # числа нет в карточке
               "Org/Base-4B": {"mmlu_pro": 41.0, "instruct": False}}
    hw = Hardware(gpu_name="X", gpu_count=1, vram_gb=24, measured_tflops=50, ram_gb=64, disk_free_gb=500)
    with llm.override_model(card_llm(answers)):
        models = asyncio.run(mh.refresh(hw, cfg))
    by_id = {m.id: m for m in models}
    assert set(by_id) == {"Org/Strong-4B", "Org/Weak-4B", "Org/Halluc-4B"}  # базовая и неразобранная отпали
    assert by_id["Org/Strong-4B"].quality == 70.5 and by_id["Org/Halluc-4B"].quality == 0.0
    assert by_id["Org/Weak-4B"].languages == ["en"] and by_id["Org/Halluc-4B"].languages == ["multi"]
    reg = Registry.load(cfg)  # каталог подхватывается реестром (и рабочим процессом обучения)
    assert reg.model("Org/Strong-4B").source == "hub" and reg.model("Qwen/Qwen3-1.7B").source == "registry"
    ru = Registry.load(cfg).filter_models(__import__("factory.core.contracts", fromlist=["Constraints"]).Constraints(), "ru")
    assert "Org/Weak-4B" not in {m.id for m in ru}  # карточка называет языки, русского среди них нет
    rec = [r for r in rank_models(ru, hw, language="ru") if r["recommended"]][0]
    assert rec["id"] == "Org/Strong-4B"  # 70.5 — лучше всего в реестре, что помещается в 24 ГБ без квантизации
    n = len(calls)
    with llm.override_model(card_llm(answers)):
        asyncio.run(mh.refresh(hw, cfg))  # свежий кэш: ни сети, ни LLM
        asyncio.run(mh.refresh(hw, cfg, force=True))  # ревизии те же: карточки из кэша
    assert len(calls) == n


def test_bnb_4bit_copies_of_official_models():
    """Готовые bnb-4bit копии берутся, если исходная модель — официальный релиз (из списков HF или реестра);
    из нескольких копий одной модели — самая скачиваемая; AWQ/GPTQ/GGUF не обучаются здесь."""
    bnb = ("bitsandbytes", "4-bit")
    listing = [
        hub_model("Org/Big-14B-Instruct", params_b=40),  # не влезет даже в 4 битах, но официальный релиз
        hub_model("unsloth/Big-14B-Instruct-bnb-4bit", params_b=8, tags=bnb, base="Org/Big-14B-Instruct", downloads=50_000),
        hub_model("unsloth/Big-14B-Instruct-unsloth-bnb-4bit", params_b=9, tags=bnb, base="Org/Big-14B-Instruct", downloads=9_000),
        hub_model("unsloth/Fan-7B-bnb-4bit", params_b=4, tags=bnb, base="Someone/Fan-7B"),  # копия чужого дообучения
        hub_model("unsloth/Known-4B-bnb-4bit", params_b=2.5, tags=bnb, base="Reg/Known-4B"),  # исходная — в реестре
        hub_model("Org/Big-14B-Instruct-AWQ", params_b=8, tags=("4-bit", "awq"), base="Org/Big-14B-Instruct"),
    ]
    api = SimpleNamespace(list_models=lambda **kw: listing)
    got = {c["id"]: c for c in mh.hub_candidates(12, api, known={"Reg/Known-4B"})}
    assert set(got) == {"unsloth/Big-14B-Instruct-bnb-4bit", "unsloth/Known-4B-bnb-4bit"}
    assert got["unsloth/Big-14B-Instruct-bnb-4bit"]["prequantized"] and got["unsloth/Big-14B-Instruct-bnb-4bit"]["base"] == "Org/Big-14B-Instruct"


def test_prequantized_model_trains_only_in_qlora(registry):
    from factory.hparams.resources import assess_model
    base = registry.models["Qwen/Qwen3-4B-Instruct-2507"]
    copy = base.model_copy(update={"id": "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit", "prequantized": True})
    hw = Hardware(gpu_name="X", gpu_count=1, vram_gb=24, measured_tflops=50, ram_gb=64, disk_free_gb=500)
    assert assess_model(base, hw)["mode"] == "lora" and assess_model(copy, hw)["mode"] == "qlora"
    assert assess_model(copy, hw)["download_gb"] < assess_model(base, hw)["download_gb"] / 3
