"""Интеграция рабочего процесса обучения на крошечной локальной модели (без скачивания весов).

🔎 из §3.4/§6: продолжение раунда не поднимает lr выше конца прошлого раунда; P-tuning через SFTTrainer работает."""
import asyncio
import json
import random
from pathlib import Path

import pandas as pd
import pytest

from factory.core.contracts import (
    DatasetCandidate, LoraParams, OptimParams, PtuningParams, Recipe, RunConfig, ScheduleParams, Seeds, SplitProportions,
)
from factory.core.storage import read_json, read_jsonl, write_json
from factory.prep.split import save_splits, split_frame
from factory.runs import create_run
from factory.training import launcher

pytestmark = pytest.mark.integration

LABELS = ["позитив", "негатив"]


@pytest.fixture
def run_paths(data_dir, tiny_registered, registry):

    def make(adapter="lora", backend=None, quantization="none", scheduler="cosine_with_min_lr", task="sentiment_analysis"):
        from factory.core.contracts import TaskSpec
        from factory.core.registry import Registry
        reg = Registry.load()
        rng = random.Random(0)
        alt = None
        if task == "ner":
            names, cities = ["Иван", "Пётр", "Анна", "Мария"], ["Москве", "Казани", "Сочи"]
            rows = []
            for i in range(160):
                n, c = rng.choice(names), rng.choice(cities)
                rows.append((f"{n} живёт в {c} {i}", f"PER: {n}; LOC: {c}") if i % 4 else (f"просто текст {i}", "нет"))
            labels = ["PER", "LOC"]
        elif task == "qa":
            rows, alt = [], []
            for i in range(160):
                c = rng.choice(["Москве", "Казани", "Сочи"])
                rows.append((f"Контекст: Музей номер {i} находится в {c}. Он открыт днём.\nВопрос: Где находится музей?", c))
                alt.append(json.dumps([c, f"в {c}"], ensure_ascii=False))
            labels = None
        else:
            rows = [(" ".join(rng.choices(["хороший", "отличный", "прекрасный", "товар"], k=5)) + f" {i}", "позитив") for i in range(120)]
            rows += [(" ".join(rng.choices(["плохой", "ужасный", "отвратительный", "товар"], k=5)) + f" {i}", "негатив") for i in range(120)]
            labels = LABELS
        spec = reg.apply_task_metrics(TaskSpec(description="Тестовая задача", task_type=task, labels=labels,
                                               base_model_id=str(tiny_registered)))
        df = pd.DataFrame(rows, columns=["input", "target"])
        if alt is not None:
            df["alt"] = alt
        df["synthetic"] = False
        df.insert(0, "_id", [str(i) for i in range(len(df))])
        session = data_dir / "sessions" / "s1"
        train, val, test = split_frame(df, SplitProportions(), seed=1, stratify=task == "sentiment_analysis")
        test_hash = save_splits(session / "data", train, val, test)
        recipe = Recipe(dataset_id="user:toy", split_seed=1, test_hash=test_hash)
        lora = LoraParams(r=4, alpha=8, target_modules=["q_proj", "v_proj"]) if adapter == "lora" else None
        pt = PtuningParams(num_virtual_tokens=4, encoder_hidden_size=16) if adapter == "ptuning" else None
        cfg = RunConfig(backend=backend or ("peft_lora" if lora else "peft_ptuning"), quantization=quantization,
                        base_model_id=str(tiny_registered), max_seq_len=96 if task == "sentiment_analysis" else 256,
                        lora=lora, ptuning=pt, optim=OptimParams(lr=5e-3 if lora else 3e-2, warmup=0.0 if scheduler == "constant" else 0.1,
                                                         scheduler=scheduler),
                        schedule=ScheduleParams(epochs=2, micro_batch=8, grad_accum=1, logging_steps=2, eval_steps=10,
                                                patience=5, best_metric=f"eval_{spec.primary_metric}"),
                        seeds=Seeds(split_seed=1, train_seed=3))
        paths, _ = create_run(data_dir / "runs", spec, DatasetCandidate(hf_id="user:toy", source="user"), recipe, cfg,
                              session, "s1")
        return paths, cfg
    return make


def run(paths, mode, status, *extra):
    res = asyncio.run(launcher.run(paths, mode, paths.root / status, *extra))
    if res.get("status") != "ok":
        print(paths.train_log.read_text(encoding="utf-8")[-4000:])
    return res


def test_lora_rounds_pilot_continue_and_final(run_paths):
    paths, cfg = run_paths("lora")
    assert run(paths, "baseline", "b.json")["status"] == "ok"
    base = read_json(paths.baselines_val)
    assert {"majority", "base_prompt"} <= set(base)

    write_json(paths.pilot / "config.json", cfg)
    pilot = run(paths, "pilot", "p.json", "--steps", "6")
    assert pilot["status"] == "ok" and pilot["sec_per_step"] > 0 and pilot["estimated_time_s"] > 0

    r1 = run(paths, "train", "rounds/round_1/status.json", "--round", "1")
    assert r1["status"] == "ok", r1
    assert (paths.round_dir(1) / "adapter_best" / "adapter_config.json").exists()
    ckpt = r1["checkpoint"]
    assert ckpt and (pd.io.common.file_exists(f"{ckpt}/optimizer.pt"))
    assert "macro_f1" in r1["val_metrics"] and "majority" in r1["baselines_val"]
    points = [p for p in read_jsonl(paths.metrics)[0]]
    assert any(p.get("train_loss") is not None for p in points)
    assert any("macro_f1" in (p.get("val_metrics") or {}) for p in points)
    lr_end_1 = r1["lr_end"]
    assert "MLflow недоступен" not in paths.train_log.read_text(encoding="utf-8")
    assert list((paths.root.parents[1] / "mlruns").glob("**/meta.yaml"))

    # 🔎 продолжение: lr второго раунда не выше, чем в конце первого
    r2 = run(paths, "train", "rounds/round_2/status.json", "--round", "2", "--epochs", "1", "--resume-from", ckpt,
             "--lr", str(lr_end_1), "--step-offset", str(r1["global_step"]), "--epoch-offset", str(r1["epochs_done"]))
    assert r2["status"] == "ok", r2
    lrs_2 = [p["lr"] for p in read_jsonl(paths.metrics)[0] if p.get("lr") is not None and p["step"] > r1["global_step"]]
    assert lrs_2 and max(lrs_2) <= lr_end_1 * (1 + 1e-6)
    steps = [p["step"] for p in read_jsonl(paths.metrics)[0] if p.get("train_loss") is not None]
    assert steps == sorted(steps)  # сквозная нумерация шагов между раундами
    assert "Состояние оптимизатора восстановлено" in paths.train_log.read_text(encoding="utf-8")

    fin = run(paths, "final_eval", "f.json", "--adapter", r2["adapter_best"])
    assert fin["status"] == "ok"
    report = read_json(paths.eval_report)
    assert report["split"] == "test" and set(report["metrics"]) == {"macro_f1", "accuracy", "weighted_f1"}
    assert any(e["type"] == "test_opened" for e in read_jsonl(paths.events)[0])


def test_retry_in_same_round_preserves_previous_checkpoints(run_paths):
    from pathlib import Path
    from factory.training.checkpoints import last_checkpoint
    paths, cfg = run_paths("lora")
    first = run(paths, "train", "first.json", "--round", "1", "--epochs", "1")
    assert first["status"] == "ok", first
    old = Path(first["checkpoint"])
    old_state = (old / "trainer_state.json").read_bytes()
    second = run(paths, "train", "second.json", "--round", "1", "--epochs", "0.2",
                 "--resume-from", str(old), "--lr", str(first["lr_end"]),
                 "--step-offset", str(first["global_step"]), "--epoch-offset", str(first["epochs_done"]))
    assert second["status"] == "ok", second
    new = Path(second["checkpoint"])
    assert new.parent != old.parent
    assert int(new.name.split("-")[-1]) < int(old.name.split("-")[-1])
    assert last_checkpoint(paths.round_dir(1)) == new
    assert (old / "trainer_state.json").read_bytes() == old_state
    assert second["global_step"] > first["global_step"]


def test_offline_worker_trains_registered_local_model(run_paths, data_dir, tiny_registered, monkeypatch):
    from factory.core.config import AppConfig
    from factory.core.local_models import register_local_model
    paths, _ = run_paths("lora")
    register_local_model(AppConfig(data_dir=data_dir, offline=True), tiny_registered, languages=["ru"])
    monkeypatch.setenv("FACTORY_OFFLINE", "1")
    result = run(paths, "train", "offline.json", "--round", "1", "--epochs", "1")
    assert result["status"] == "ok", result
    assert result["checkpoint"] and Path(result["checkpoint"]).exists()


def test_ptuning_through_sfttrainer(run_paths):
    paths, cfg = run_paths("ptuning")
    r1 = run(paths, "train", "rounds/round_1/status.json", "--round", "1", "--epochs", "1")
    assert r1["status"] == "ok", r1
    adapter_cfg = json.loads((paths.round_dir(1) / "adapter_best" / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter_cfg["peft_type"] == "P_TUNING"


def test_control_stop_and_save(run_paths):
    from factory.training.supervisor import send_command
    paths, cfg = run_paths("lora")
    send_command(paths, "stop_and_save")  # команда уже ждёт: процесс остановится на первом шаге
    r1 = run(paths, "train", "rounds/round_1/status.json", "--round", "1")
    assert r1["status"] == "ok" and r1["global_step"] <= 2 and r1["checkpoint"]


def test_qlora_pilot_4bit(run_paths):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("bitsandbytes 4-bit требует CUDA")
    paths, cfg = run_paths("lora")
    q = cfg.model_copy(update={"quantization": "4bit"})
    write_json(paths.pilot / "config.json", q)
    res = run(paths, "pilot", "p.json", "--steps", "4")
    assert res["status"] == "ok", res


def test_eval_truncation_keeps_prompt_tail(tiny_model):
    from factory.training.formatting import prompt_builder
    from factory.training.worker import load_tokenizer, truncate_inputs
    tok = load_tokenizer(str(tiny_model))
    long_input = " ".join(["хороший отличный товар"] * 300)
    build = prompt_builder("Тональность. Ответь одной меткой из списка: позитив, негатив.")
    [cut] = truncate_inputs([long_input], None, tok, build, max_seq_len=96, reserve=8)
    prompt = build(cut)
    assert prompt.endswith("Target:")
    assert len(tok(prompt)["input_ids"]) + 8 <= 96


def test_unsloth_lora_round_continue_and_final(run_paths):
    """Unsloth: раунд, продолжение с чекпоинта (веса адаптера и оптимизатор), финальная оценка."""
    paths, cfg = run_paths("lora", backend="unsloth_lora")
    r1 = run(paths, "train", "rounds/round_1/status.json", "--round", "1", "--epochs", "1")
    assert r1["status"] == "ok", r1
    assert "Unsloth" in paths.train_log.read_text(encoding="utf-8")
    r2 = run(paths, "train", "rounds/round_2/status.json", "--round", "2", "--epochs", "1", "--resume-from", r1["checkpoint"],
             "--lr", str(r1["lr_end"]), "--step-offset", str(r1["global_step"]), "--epoch-offset", str(r1["epochs_done"]))
    assert r2["status"] == "ok", r2
    fin = run(paths, "final_eval", "f.json", "--adapter", r2["adapter_best"], "--round", "2")
    assert fin["status"] == "ok", fin
    assert read_json(paths.eval_report)["latency_ms"] > 0


def test_unsloth_qlora_pilot(run_paths):
    paths, cfg = run_paths("lora", backend="unsloth_lora", quantization="4bit")
    write_json(paths.pilot / "config.json", cfg)
    res = run(paths, "pilot", "p.json", "--steps", "4")
    assert res["status"] == "ok", res


def test_monitor_agent_controls_live_training(run_paths, cfg, registry):
    """A4 в реальном времени: плановая проверка → scale_lr и eval_now применяются живым процессом."""
    from pydantic_ai import ModelResponse, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import FunctionModel

    from factory.core import llm
    from factory.core.contracts import SessionSettings
    from factory.training.supervisor import RoundPlan, Supervisor

    paths, run_cfg = run_paths("lora", scheduler="constant")
    state = {"intervened": False}

    def a4(messages, info):
        returned = [p.tool_name for m in messages for p in getattr(m, "parts", []) if isinstance(p, ToolReturnPart)]
        out = next(t.name for t in info.output_tools)
        if state["intervened"] or "request_eval_now" in returned:
            state["intervened"] = True
            return ModelResponse(parts=[ToolCallPart(tool_name=out, args={"verdict": "ok", "explanation": "lr снижен"})])
        if "scale_lr" in returned:
            return ModelResponse(parts=[ToolCallPart(tool_name="request_eval_now", args={"reason": "проверить"})])
        return ModelResponse(parts=[ToolCallPart(tool_name="scale_lr", args={"factor": 0.5, "reason": "тест"})])

    cfg.training.poll_interval_s = 0.3
    cfg.training.monitor_cooldown_steps = 0
    cfg.training.monitor_check_every_evals = 1
    sup = Supervisor(paths, cfg, registry, SessionSettings(), notify=lambda t: None, pilot_tps=None, vram_gb=None)
    with llm.override_model(FunctionModel(a4)):
        out = asyncio.run(sup.run_round(RoundPlan(round_k=1, epochs=3), run_cfg))
    assert out.status == "ok", out.result
    events = read_jsonl(paths.events)[0]
    applied = {e["cmd"]: e["step"] for e in events if e["type"] == "control_applied"}
    assert {"patch", "eval_now"} <= set(applied)
    lrs = [(p["step"], p["lr"]) for p in read_jsonl(paths.metrics)[0] if p.get("lr") is not None]
    before = [lr for step, lr in lrs if step < applied["patch"]]
    # лог transformers показывает lr шедулера с задержкой в одну точку; оптимизатор получает новый lr сразу
    after = [lr for step, lr in lrs if step > applied["patch"] + 1]
    assert before and after and max(after) == pytest.approx(min(before) * 0.5)
    review = next(e for e in events if e["type"] == "monitor_review")
    assert [a["action"] for a in review["actions"]] == ["patch", "eval_now"]


@pytest.mark.parametrize("task,primary,baseline", [("ner", "entity_f1", "gazetteer"), ("qa", "token_f1", "best_sentence")])
def test_ner_and_qa_train_and_final_eval(run_paths, task, primary, baseline):
    """NER и QA целиком через рабочий процесс: обучение, оценка на val с метрикой задачи, test, baseline'ы."""
    paths, cfg = run_paths("lora", task=task)
    r1 = run(paths, "train", "rounds/round_1/status.json", "--round", "1", "--epochs", "1")
    assert r1["status"] == "ok", r1
    assert primary in r1["val_metrics"] and baseline in r1["baselines_val"]
    assert any(primary in (p.get("val_metrics") or {}) for p in read_jsonl(paths.metrics)[0])  # лучший чекпоинт — по метрике задачи
    fin = run(paths, "final_eval", "f.json", "--adapter", r1["adapter_best"], "--round", "1")
    assert fin["status"] == "ok", fin
    report = read_json(paths.eval_report)
    assert primary in report["metrics"] and baseline in report["baselines"] and "base_prompt" in report["baselines"]
    if task == "qa":  # «в Москве» и «Москве» — оба эталона, лучший засчитывается
        assert report["baselines"]["best_sentence"]["token_f1"] > 0
    if task == "ner":
        assert report["baselines"]["gazetteer"]["entity_f1"] > 0.5  # словарь из train почти идеально находит сущности
