"""Guard, ресурсы, эвристика (обёртка курсовой), пилотные авто-исправления, история запусков."""
import pytest

from factory.core.contracts import (
    DataProfile, LengthStats, LoraParams, OptimParams, ParamRationale, RunConfig, ScheduleParams, Seeds,
)
from factory.hparams import guard, heuristic, history, pilot, resources


@pytest.fixture
def ctx(cfg, registry, cls_spec):
    profile = DataProfile(n_rows=2000, input_tokens_est=LengthStats(p95=80, p99=120),
                          target_tokens_est=LengthStats(p95=3, p99=4))
    return guard.HParamContext(spec=cls_spec, task=registry.task(cls_spec.task_type),
                               model=registry.model("Qwen/Qwen3-0.6B"), profile=profile, n_train=2000,
                               vram_gb=12.0, cfg=cfg, seeds=Seeds(split_seed=1, train_seed=2))


def test_guard_clamps_and_takes_modules_from_registry(ctx):
    choice = guard.StrategyChoice(lr=0.5, epochs=100, lora=guard.LoraChoice(r=1000, alpha=1, modules="all_linear"))
    cfg = guard.assemble(choice, ctx, "agent")
    assert cfg.optim.lr == ctx.cfg.hparams.ranges.lr[1]
    assert cfg.schedule.epochs == ctx.cfg.hparams.ranges.epochs[1]
    assert cfg.lora.r == ctx.cfg.hparams.ranges.lora_r[1]
    assert cfg.lora.alpha >= cfg.lora.r * ctx.cfg.hparams.ranges.lora_alpha_ratio[0]
    assert cfg.lora.target_modules == ctx.model.lora_modules["all_linear"]
    assert any("guard" in n for n in ctx.notes)
    assert cfg.schedule.best_metric == "eval_macro_f1"


def test_guard_quantization_by_vram(ctx):
    choice = guard.StrategyChoice(lr=1e-4, epochs=3, lora=guard.LoraChoice(r=8, alpha=16))
    assert guard.assemble(choice, ctx, "heuristic").quantization == "none"
    big = guard.HParamContext(**{**ctx.__dict__, "model": ctx.model.model_copy(update={"params_b": 8.0}), "notes": []})
    assert guard.assemble(choice, big, "heuristic").quantization == "4bit"


def test_guard_user_qlora_forces_4bit(ctx):
    spec = ctx.spec.model_copy(update={"constraints": ctx.spec.constraints.model_copy(update={"adapter_method": "qlora"})})
    c = guard.HParamContext(**{**ctx.__dict__, "spec": spec, "notes": []})
    cfg = guard.assemble(guard.StrategyChoice(lr=1e-4, epochs=3, lora=guard.LoraChoice(r=8, alpha=16)), c, "heuristic")
    assert cfg.quantization == "4bit"


def test_guard_ptuning(ctx):
    spec = ctx.spec.model_copy(update={"constraints": ctx.spec.constraints.model_copy(update={"adapter_method": "ptuning"})})
    c = guard.HParamContext(**{**ctx.__dict__, "spec": spec, "notes": []})
    choice = guard.StrategyChoice(lr=1e-3, epochs=3, ptuning=guard.PtuningChoice(num_virtual_tokens=1000, encoder_hidden_size=99999))
    cfg = guard.assemble(choice, c, "agent")
    assert cfg.backend == "peft_ptuning" and cfg.lora is None
    assert cfg.ptuning.num_virtual_tokens == ctx.cfg.hparams.ranges.num_virtual_tokens[1]
    assert cfg.ptuning.encoder_hidden_size <= ctx.model.hidden_size


def test_validate_user_config_rejects_foreign_modules(ctx):
    cfg = guard.assemble(guard.StrategyChoice(lr=1e-4, epochs=3, lora=guard.LoraChoice(r=8, alpha=16)), ctx, "heuristic")
    bad = cfg.model_copy(update={"lora": cfg.lora.model_copy(update={"target_modules": ["evil_proj"]})})
    assert any("target_modules" in p for p in guard.validate_user_config(bad, ctx))
    assert guard.validate_user_config(cfg, ctx) == []


def test_runconfig_requires_exactly_one_adapter():
    base = dict(backend="peft_lora", base_model_id="m", max_seq_len=128, optim=OptimParams(lr=1e-4),
                schedule=ScheduleParams(epochs=1), seeds=Seeds(split_seed=1, train_seed=1))
    with pytest.raises(ValueError):
        RunConfig(**base)
    RunConfig(**base, lora=LoraParams(r=8, alpha=16))


def test_max_seq_len_and_eval_steps(registry):
    m = registry.model("Qwen/Qwen3-0.6B")
    assert resources.choose_max_seq_len(100, 10, m, 2048) % 64 == 0
    assert resources.choose_max_seq_len(10_000, 10_000, m, 2048) == 2048
    log_s, eval_s = resources.logging_and_eval_steps(n_train=200, micro_batch=4, grad_accum=4)
    assert eval_s <= max(1, 13 // 2)  # не реже двух раз за эпоху
    assert resources.logging_and_eval_steps(1_000_000, 4, 4) == (50, 400)


def test_heuristic_lora_lr_does_not_fall_with_data(ctx):
    """lr LoRA — 2e-4 на любом объёме данных (раньше правила курсовой давали 5e-5 на больших датасетах)."""
    for n_train in (300, 5_000, 30_000, 200_000):
        c = guard.HParamContext(**{**ctx.__dict__, "n_train": n_train, "notes": []})
        assert heuristic.suggest(c, None).lr == pytest.approx(2e-4), n_train
    small = guard.HParamContext(**{**ctx.__dict__, "n_train": 300, "notes": []})
    choice = heuristic.suggest(small, baseline_primary=None)
    assert (choice.lora.r, choice.lora.alpha, choice.lora.modules) == (8, 16, "all_linear")
    assert heuristic.suggest(small, baseline_primary=0.9).epochs == choice.epochs - 1
    assert heuristic.suggest(small, baseline_primary=0.2).epochs == choice.epochs + 1


def test_pilot_fixes():
    cfg = RunConfig(backend="peft_lora", base_model_id="m", max_seq_len=128, lora=LoraParams(r=8, alpha=16),
                    optim=OptimParams(lr=2e-4, warmup=0.03), schedule=ScheduleParams(epochs=1, micro_batch=4, grad_accum=4),
                    seeds=Seeds(split_seed=1, train_seed=1))
    c1, _ = pilot.fix_oom(cfg)
    assert (c1.schedule.micro_batch, c1.schedule.grad_accum) == (2, 8)  # эффективный батч тот же
    c = cfg.model_copy(update={"schedule": cfg.schedule.model_copy(update={"micro_batch": 1})})
    c2, _ = pilot.fix_oom(c)
    assert c2.schedule.gradient_checkpointing
    c3, _ = pilot.fix_oom(c2)
    assert c3.quantization == "4bit"
    assert pilot.fix_oom(c3) is None
    n, _ = pilot.fix_nan(cfg)
    assert n.optim.lr == pytest.approx(1e-4) and n.optim.warmup > cfg.optim.warmup


def test_doc_lookup(cfg):
    hits = history.lookup_docs(cfg.hparam_docs_dir, "learning rate lora")
    assert hits and all("#" in h["doc_chunk_id"] for h in hits)


def test_rationale_model():
    r = ParamRationale(param="lr", value=1e-4, reason="x")
    assert r.evidence == []


def test_heuristic_epochs_fall_with_dataset_size(ctx):
    """Больше данных — меньше эпох (1–3 по практике LoRA); больше ранг."""
    got = [heuristic.suggest(guard.HParamContext(**{**ctx.__dict__, "n_train": n, "notes": []}), None)
           for n in (300, 2000, 30_000, 200_000)]
    assert [g.epochs for g in got] == [4, 3, 2, 1] and [g.lora.r for g in got] == [8, 16, 32, 32]


def test_gpu_hours_budget_cuts_epochs(ctx):
    from factory.hparams.pilot import PilotReport
    from factory.hparams.strategy import fit_budget
    spec = ctx.spec.model_copy(update={"constraints": ctx.spec.constraints.model_copy(update={"gpu_hours": 1.0})})
    c = guard.HParamContext(**{**ctx.__dict__, "spec": spec, "notes": []})
    cfg = guard.assemble(guard.StrategyChoice(lr=1e-4, epochs=6, lora=guard.LoraChoice(r=8, alpha=16)), c, "heuristic")
    report = PilotReport(status="ok", config=cfg, result={"status": "ok", "estimated_time_s": 3 * 3600})
    fitted = fit_budget(report, c)
    assert fitted.schedule.epochs == 2 and report.result["estimated_time_s"] == 3600
    assert any("бюджет" in n for n in c.notes)


def test_gpu_hours_budget_below_one_epoch_caps_steps(ctx):
    """Бюджет меньше одной эпохи: не «предупредить и учить дольше», а ограничить первый раунд шагами."""
    from factory.hparams.pilot import PilotReport
    from factory.hparams.strategy import fit_budget
    spec = ctx.spec.model_copy(update={"constraints": ctx.spec.constraints.model_copy(update={"gpu_hours": 1.0})})
    c = guard.HParamContext(**{**ctx.__dict__, "spec": spec, "notes": []})
    cfg = guard.assemble(guard.StrategyChoice(lr=1e-4, epochs=2, lora=guard.LoraChoice(r=8, alpha=16)), c, "heuristic")
    report = PilotReport(status="ok", config=cfg, result={"status": "ok", "estimated_time_s": 8 * 3600, "steps_per_epoch": 1000})
    fitted = fit_budget(report, c)
    assert fitted.schedule.epochs == 1 and fitted.schedule.max_steps == 250  # эпоха ~4 ч, бюджет 1 ч → четверть
    assert report.result["estimated_time_s"] == 3600 and any("250 шагами" in n for n in c.notes)


def test_a3_instructions_include_task_heuristics(ctx):
    from factory.agents.hparam import SIZE_HEURISTICS, TASK_HEURISTICS
    assert set(TASK_HEURISTICS) >= {"text_classification", "sentiment_analysis", "summarization", "text_generation", "ner", "qa"}
    assert next(h for limit, h in SIZE_HEURISTICS if 500 < limit).startswith("мало данных")
