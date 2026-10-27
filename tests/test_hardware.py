"""Подбор моделей под железо: режим LoRA/QLoRA/не влезает, RAM, диск, время, рекомендация."""
import pytest

from factory.core.contracts import Constraints
from factory.core.hardware import Hardware
from factory.hparams.resources import assess_model, rank_models


def hw(vram, ram=64, disk=500, tflops=50.0):
    return Hardware(gpu_name="GPU", gpu_count=1 if vram else 0, vram_gb=vram, bf16=True, measured_tflops=tflops,
                    ram_gb=ram, ram_available_gb=ram, disk_free_gb=disk)


def test_modes_on_8gb_card(registry):
    m = registry.models
    machine = hw(8)
    assert assess_model(m["Qwen/Qwen3-0.6B"], machine)["mode"] == "lora"
    assert assess_model(m["Qwen/Qwen2.5-7B-Instruct"], machine)["mode"] == "qlora"
    assert assess_model(m["Qwen/Qwen3-8B"], machine)["mode"] == "none"  # с KV-кэшем оценки на val уже не влезает
    big = assess_model(m["Qwen/Qwen3-14B"], machine)
    assert big["mode"] == "none" and not big["fits"] and "VRAM" in big["problems"][0]


def test_everything_fits_on_80gb(registry):
    rows = rank_models(list(registry.models.values()), hw(80))
    assert all(r["mode"] == "lora" for r in rows)


def test_no_gpu_means_no_training(registry):
    r = assess_model(registry.models["Qwen/Qwen3-0.6B"], hw(0))
    assert not r["fits"] and "GPU" in r["problems"][0]


def test_ram_and_disk_limits(registry):
    r = assess_model(registry.models["Qwen/Qwen3-8B"], hw(80, ram=8, disk=10))
    assert not r["ram_ok"] and not r["disk_ok"] and not r["fits"]


def test_time_estimate_scales_with_model_and_speed(registry):
    small, big = registry.models["Qwen/Qwen3-0.6B"], registry.models["Qwen/Qwen3-4B"]
    assert assess_model(big, hw(80))["hours_typical"] > assess_model(small, hw(80))["hours_typical"]
    assert assess_model(small, hw(80, tflops=100))["hours_typical"] < assess_model(small, hw(80, tflops=25))["hours_typical"]


def test_recommendation_prefers_best_quality_within_budget_and_language(registry):
    models = list(registry.models.values())
    rows = rank_models(models, hw(24), budget_hours=1000, language="ru")
    rec = [r for r in rows if r["recommended"]]
    assert len(rec) == 1 and rec[0]["mode"] == "lora" and rec[0]["native_language"]
    assert rec[0]["quality"] == max(r["quality"] for r in rows if r["mode"] == "lora" and r["native_language"])
    tight = [r for r in rank_models(models, hw(24), budget_hours=0.5, language="ru") if r["recommended"]][0]
    assert tight["params_b"] < rec[0]["params_b"]  # маленький бюджет времени — модель меньше


def test_recommendation_prefers_newer_generation_over_size(registry):
    """Новое поколение меньшего размера сильнее старого: Qwen3.5-2B (MMLU-Pro 55) лучше Qwen3-1.7B (40)."""
    models = [registry.models[i] for i in ("Qwen/Qwen3-1.7B", "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B")]
    rec = [r for r in rank_models(models, hw(24), budget_hours=1000, language="ru") if r["recommended"]]
    assert rec[0]["id"] == "Qwen/Qwen3.5-2B"


def test_available_models_filters_by_hardware_and_method(registry):
    machine = hw(8)
    ids = {r["id"] for r in registry.available_models(Constraints(), "ru", machine)}
    assert "Qwen/Qwen2.5-7B-Instruct" in ids and "Qwen/Qwen3-14B" not in ids
    lora_only = {r["id"] for r in registry.available_models(Constraints(adapter_method="lora"), "ru", machine)}
    assert "Qwen/Qwen2.5-7B-Instruct" not in lora_only  # явный выбор LoRA без квантизации — QLoRA-only модели не подходят


def test_finalize_spec_rejects_model_that_does_not_fit(registry):
    from factory.agents.intake import SpecDraft, finalize_spec
    draft = SpecDraft(description="d", task_type="sentiment_analysis", labels=["a", "b"], base_model_id="Qwen/Qwen3-14B")
    with pytest.raises(ValueError, match="железо"):
        finalize_spec(draft, registry, hw(8))
    assert finalize_spec(draft.model_copy(update={"base_model_id": "Qwen/Qwen3-1.7B"}), registry, hw(8)).base_model_id


def test_detect_runs_in_subprocess_and_caches(tmp_path):
    from factory.core import hardware
    h = hardware.detect(tmp_path)
    assert h.ram_gb > 0 and h.cpu_count >= 1 and (tmp_path / "cache" / "hardware.json").exists()
    assert hardware.detect(tmp_path) is h  # кэш в процессе


def test_memory_taken_by_other_programs_is_not_planned_for(registry):
    """Браузеры и редакторы держат часть VRAM: на Windows перебор не даёт ошибки, а замедляет обучение в разы —
    модель подбирается под оставшееся."""
    models = list(registry.models.values())
    free = [r for r in rank_models(models, hw(12), language="ru") if r["recommended"]][0]
    busy_hw = hw(12).model_copy(update={"vram_others_gb": 5.0})
    assert busy_hw.usable_vram_gb == 7.0
    busy = [r for r in rank_models(models, busy_hw, language="ru") if r["recommended"]][0]
    assert (busy["id"], busy["mode"]) != (free["id"], free["mode"])
    assert busy["mode"] == "qlora" or busy["params_b"] < free["params_b"]
