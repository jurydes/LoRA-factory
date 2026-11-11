"""Рабочий процесс обучения (отдельный процесс; закрытая вкладка его не останавливает).

Режимы:
  baseline   — baseline'ы на val (простые + исходная модель с промптом), до подбора гиперпараметров;
  pilot      — короткий прогон: NaN/OOM, токены/с, пиковая память, оценка времени;
  train      — раунд обучения: чекпоинт, лучший адаптер, val-метрики → rounds/round_k/status.json;
  final_eval — один раз при окончательной остановке: test открывается, eval_report.json.

Запуск: python -m factory.training.worker --run-dir RUN --mode train --round 1 [--resume-from CKPT --lr LR] [--epochs N]
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
if os.name == "nt":
    # Unsloth + flex attention на Windows (triton-windows): на Gemma3 оценка генерацией посреди обучения падает
    # с «CUDA illegal memory access», а внимание перетрассируется dynamo почти на каждом шаге — ~500 токенов/с
    # против ~3000 с SDPA (RTX 3060, gemma-3-1b). До импорта unsloth: флаг читается при импорте.
    os.environ.setdefault("UNSLOTH_ENABLE_FLEX_ATTENTION", "0")

from factory.core.config import get_config
from factory.core.contracts import Manifest, MetricPoint, Recipe, RunConfig
from factory.core.registry import Registry
from factory.core.storage import (
    append_jsonl,
    file_lock,
    read_json,
    read_model,
    write_json,
)
from factory.runs import RunPaths

logger = logging.getLogger("factory.worker")

BEST_EVAL_SAMPLES_DURING_TRAINING = 128


# ---------------------------------------------------------------- загрузка

def _dtype():
    import torch
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    if get_config().offline and not Path(model_id).is_dir():
        raise RuntimeError(f"В закрытом контуре модель должна быть локальным каталогом: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=get_config().offline)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _hf_login():
    if get_config().offline:
        return
    try:
        from settings import load_global_config
        token = load_global_config().hf_token
        if token:
            from huggingface_hub import login
            login(token=token)
    except Exception as e:
        logger.debug("HF login пропущен: %s", e)


def load_unsloth(name: str, cfg: RunConfig):
    """Unsloth: LoRA в bf16 или QLoRA в 4 битах (быстрые ядра, меньше памяти)."""
    from unsloth import FastLanguageModel
    if get_config().offline and not Path(name).is_dir():
        raise RuntimeError(f"В закрытом контуре модель должна быть локальным каталогом: {name}")
    info = Registry.load().models.get(cfg.base_model_id)
    extra = {"text_only": True} if info and info.text_only else {}  # мультимодальный чекпоинт: без vision-части
    model, tok = FastLanguageModel.from_pretrained(
        model_name=name, max_seq_length=cfg.max_seq_len, load_in_4bit=cfg.quantization == "4bit", dtype=None, **extra)
    tok = getattr(tok, "tokenizer", tok)  # processor мультимодальной модели → его токенизатор
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def load_base_model(cfg: RunConfig):
    """База (с квантизацией при необходимости) + токенизатор. Unsloth — только для unsloth_lora."""
    import torch
    if get_config().offline and not Path(cfg.base_model_id).is_dir():
        raise RuntimeError(f"В закрытом контуре модель должна быть локальным каталогом: {cfg.base_model_id}")
    _hf_login()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.backend == "unsloth_lora":
        return load_unsloth(cfg.base_model_id, cfg)
    from transformers import AutoModelForCausalLM
    kwargs = dict(dtype=_dtype(), device_map={"": device}, attn_implementation="sdpa")
    if cfg.quantization == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_dtype())
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, local_files_only=get_config().offline, **kwargs)
    if cfg.quantization == "4bit":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.schedule.gradient_checkpointing)
    return model, load_tokenizer(cfg.base_model_id)


def peft_config(cfg: RunConfig):
    from peft import LoraConfig, PromptEncoderConfig
    if cfg.ptuning:
        p = cfg.ptuning
        return PromptEncoderConfig(
            task_type="CAUSAL_LM", num_virtual_tokens=p.num_virtual_tokens, encoder_hidden_size=p.encoder_hidden_size,
            encoder_reparameterization_type=p.encoder_reparameterization_type, encoder_num_layers=p.encoder_num_layers,
            encoder_dropout=p.encoder_dropout)
    l = cfg.lora
    return LoraConfig(r=l.r, lora_alpha=l.alpha, lora_dropout=l.dropout, target_modules=l.target_modules,
                      use_rslora=l.variant == "rslora", use_dora=l.variant == "dora", bias="none", task_type="CAUSAL_LM")


def build_model(cfg: RunConfig, resume_adapter: Path | None = None):
    """Модель с адаптером: новый адаптер или продолжение с адаптера из чекпоинта."""
    model, tok = load_base_model(cfg)
    if cfg.backend == "unsloth_lora":
        from unsloth import FastLanguageModel
        l = cfg.lora
        model = FastLanguageModel.get_peft_model(
            model, r=l.r, target_modules=l.target_modules, lora_alpha=l.alpha, lora_dropout=l.dropout, bias="none",
            use_rslora=l.variant == "rslora", random_state=cfg.seeds.train_seed, max_seq_length=cfg.max_seq_len,
            use_gradient_checkpointing="unsloth" if cfg.schedule.gradient_checkpointing else False)
        if resume_adapter is not None:  # продолжение раунда: веса адаптера из чекпоинта в ту же структуру
            from peft.utils import load_peft_weights, set_peft_model_state_dict
            set_peft_model_state_dict(model, load_peft_weights(str(resume_adapter)))
        return model, tok
    from peft import PeftModel, get_peft_model
    if resume_adapter is not None:
        model = PeftModel.from_pretrained(model, str(resume_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, peft_config(cfg))
    if cfg.schedule.gradient_checkpointing and cfg.quantization != "4bit" and cfg.lora:
        model.enable_input_require_grads()
    return model, tok


# ---------------------------------------------------------------- данные

def truncate_inputs(inputs: list[str], targets: list[str] | None, tok, prompt, max_seq_len: int,
                    reserve: int | None = None) -> list[str]:
    """Обрезает вход так, чтобы промпт + ответ поместились в max_seq_len (ответ не обрезается).
    reserve — сколько токенов оставить под ответ, если эталона нет (генерация при оценке)."""
    from factory.training.formatting import completion
    overhead = len(tok(prompt(""))["input_ids"]) + 2
    out = []
    in_ids = tok(inputs, add_special_tokens=False)["input_ids"]
    if reserve is not None:
        tgt_ids = [[0] * reserve] * len(inputs)
    else:
        tgt_ids = tok([completion(t) for t in targets], add_special_tokens=False)["input_ids"]
    for text, ids, t_ids in zip(inputs, in_ids, tgt_ids):
        budget = max(16, max_seq_len - overhead - len(t_ids))
        out.append(text if len(ids) <= budget else tok.decode(ids[:budget], skip_special_tokens=True))
    return out


def sft_dataset(df, tok, prompt, max_seq_len: int):
    """Записи prompt/completion. Если промпт (инструкция) с ответом не влезает в max_seq_len — явная ошибка:
    TRL молча выбросил бы такие примеры и обучение получило бы пустой или урезанный датасет."""
    from datasets import Dataset

    from factory.training.formatting import completion, to_sft_records
    inputs = truncate_inputs(df["input"].tolist(), df["target"].tolist(), tok, prompt, max_seq_len)
    records = to_sft_records(inputs, df["target"].tolist(), prompt)
    lens = [len(x) for x in tok([r["prompt"] + r["completion"] for r in records[:500]])["input_ids"]]
    too_long = sum(n > max_seq_len for n in lens) / max(len(lens), 1)
    if too_long > 0.05:
        overhead = len(tok(prompt(""))["input_ids"])
        raise ValueError(f"max_seq_len={max_seq_len} мал: {too_long:.0%} примеров не помещаются даже с обрезанным входом "
                         f"(инструкция занимает {overhead} токенов, ответы — до "
                         f"{max(len(tok(completion(t))['input_ids']) for t in df['target'].tolist()[:500])})")
    return Dataset.from_list(records)


def max_new_tokens_for(fmt: str, tok, targets: list[str], labels: list[str] | None, cap: int) -> int:
    if fmt == "label":
        return max(len(tok(l, add_special_tokens=False)["input_ids"]) for l in (labels or ["x"])) + 4
    lens = sorted(len(tok(t, add_special_tokens=False)["input_ids"]) for t in targets[:500]) or [cap]
    p95 = lens[int(0.95 * (len(lens) - 1))]
    return int(min(cap, max(16, p95 * 1.3)))


def make_evaluator(ctx: "Ctx", df, tok, n: int, cfg: RunConfig, label: str = "оценка на val",
                   token_budget: int | None = None):
    """Длина генерации — по длине ответов в данных; время оценки держится числом примеров: не больше
    token_budget сгенерированных токенов за проход (по умолчанию eval_token_budget), но не меньше min_eval_samples."""
    from factory.training.evaluator import TaskEvaluator
    from factory.training.metrics import parse_alternatives
    labels = ctx.manifest.spec.labels
    train = ctx.app.training
    new_tokens = max_new_tokens_for(ctx.task.format, tok, df["target"].tolist(), labels, train.max_new_tokens)
    n = min(n, max(train.min_eval_samples, (token_budget or train.eval_token_budget) // new_tokens))
    sub = df.sample(n=min(n, len(df)), random_state=cfg.seeds.split_seed) if len(df) > n else df
    prompt = ctx.prompt(tok)
    # вход режется заранее и так же, как при обучении (по длине эталонного ответа): обрезка токенизатором
    # справа срезала бы хвост промпта
    inputs = truncate_inputs(sub["input"].tolist(), sub["target"].tolist(), tok, prompt, cfg.max_seq_len)
    if ctx.max_context:  # промпт и сгенерированный ответ вместе — в пределах контекста модели
        inputs = truncate_inputs(inputs, None, tok, prompt, ctx.max_context, reserve=new_tokens)
    return TaskEvaluator(
        fmt=ctx.task.format, metric_names=list(ctx.task.metrics), prompt=prompt, inputs=inputs,
        references=sub["target"].tolist(), labels=labels, max_new_tokens=new_tokens,
        raw_inputs=sub["input"].tolist(),
        alternatives=parse_alternatives(sub["alt"].tolist()) if "alt" in sub else None,
        unsloth=cfg.backend == "unsloth_lora", on_progress=lambda done, total: ctx.progress(f"{label}: {done} из {total}"))


def tokens_per_step(ds, tok, cfg: RunConfig) -> float:
    """Средняя длина примера × эффективный батч — для оценки токенов/с."""
    sample = ds.select(range(min(200, len(ds))))
    avg = sum(len(x) for x in tok([r["prompt"] + r["completion"] for r in sample])["input_ids"]) / max(len(sample), 1)
    return avg * cfg.schedule.micro_batch * cfg.schedule.grad_accum


# ---------------------------------------------------------------- колбэки

def _finite(x) -> bool:
    return x is not None and math.isfinite(float(x))


def make_callbacks(paths: RunPaths, step_offset: int, epoch_offset: float, tokens_per_step: float, pilot: bool):
    import torch
    from transformers import TrainerCallback

    # пилот пишет в свою папку: его шаги не должны попадать в кривые и детекторы запуска
    metrics_path = paths.pilot / "metrics.jsonl" if pilot else paths.metrics

    class Recorder(TrainerCallback):
        """Пишет MetricPoint в metrics.jsonl, обновляет heartbeat, ловит NaN."""

        def __init__(self):
            self.nan = False
            self.last_time = time.monotonic()
            self.last_step = 0
            self.last_beat = 0.0
            self.step_times: list[float] = []
            self.mlflow = None
            self.started = time.monotonic()

        def on_train_begin(self, args, state, control, **kw):
            self.started = time.monotonic()

        def on_train_end(self, args, state, control, **kw):
            if not pilot:  # дальше — итоговая оценка раунда: команды обучению (lr, клиппинг…) уже не применятся
                append_jsonl(paths.events, {"type": "train_loop_end", "step": step_offset + state.global_step})

        def on_step_end(self, args, state, control, **kw):
            now = time.monotonic()
            self.step_times.append(now)
            if now - self.last_beat > 5 and not pilot:
                paths.heartbeat.write_text(str(time.time()))
                self.last_beat = now
            if not pilot and state.max_steps and state.global_step >= state.max_steps:
                control.should_evaluate = True  # метрики конца раунда — для отчёта и чекпоинта

        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            if "loss" not in logs:
                return
            now = time.monotonic()
            steps = max(state.global_step - self.last_step, 1)
            tps = tokens_per_step * steps / max(now - self.last_time, 1e-6)
            self.last_time, self.last_step = now, state.global_step
            loss = logs.get("loss")
            # темп с начала раунда: оценки на val идут равномерно и входят в него сами
            eta = ((now - self.started) / state.global_step * (state.max_steps - state.global_step)
                   if state.global_step and state.max_steps else None)
            point = MetricPoint(
                step=step_offset + state.global_step, epoch=round(epoch_offset + (state.epoch or 0), 4),
                train_loss=loss if _finite(loss) else float("nan"), lr=logs.get("learning_rate"),
                grad_norm=logs.get("grad_norm"), tokens_per_sec=round(tps, 1),
                gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3) if torch.cuda.is_available() else None,
                max_step=step_offset + state.max_steps if state.max_steps else None,
                eta_s=round(eta, 1) if eta is not None else None)
            append_jsonl(metrics_path, point)
            _mlflow_log(self, {"train_loss": loss, "lr": logs.get("learning_rate"), "grad_norm": logs.get("grad_norm"),
                               "tokens_per_sec": tps}, point.step)
            if not _finite(loss) or (logs.get("grad_norm") is not None and not _finite(logs["grad_norm"])):
                self.nan = True
                control.should_training_stop = True

        def record_eval(self, state, metrics: dict, pred_share: dict | None):
            val = {k[5:]: float(v) for k, v in metrics.items()
                   if k.startswith("eval_") and k not in ("eval_loss", "eval_runtime", "eval_samples_per_second",
                                                          "eval_steps_per_second", "eval_entropy", "eval_num_tokens",
                                                          "eval_mean_token_accuracy") and isinstance(v, (int, float))}
            point = MetricPoint(step=step_offset + state.global_step, epoch=round(epoch_offset + (state.epoch or 0), 4),
                                val_loss=metrics.get("eval_loss"), val_metrics=val, pred_label_share=pred_share)
            append_jsonl(metrics_path, point)
            self.last_time, self.last_step = time.monotonic(), state.global_step
            _mlflow_log(self, {"val_loss": metrics.get("eval_loss"), **{f"val_{k}": v for k, v in val.items()}}, point.step)

    class Control(TrainerCallback):
        """Команды из control.json: stop_and_save, abort, patch (lr_scale / grad_clip)."""

        def __init__(self):
            self.done: set[str] = set()
            self.aborted = False
            self.mtime = 0.0

        def on_step_end(self, args, state, control, optimizer=None, lr_scheduler=None, **kw):
            try:
                mtime = paths.control.stat().st_mtime
            except FileNotFoundError:
                return
            if mtime == self.mtime:
                return
            self.mtime = mtime
            for cmd in (read_json(paths.control, {}) or {}).get("commands", []):
                if cmd.get("id") in self.done:
                    continue
                self.done.add(cmd.get("id"))
                kind = cmd.get("cmd")
                if kind == "stop_and_save":
                    control.should_save = True
                    control.should_evaluate = True
                    control.should_training_stop = True
                elif kind == "abort":
                    self.aborted = True
                    control.should_training_stop = True
                elif kind == "eval_now":
                    control.should_evaluate = True
                elif kind == "save_now":
                    control.should_save = True
                elif kind == "patch":
                    patch = cmd.get("patch", {})
                    if patch.get("kind") == "lr_scale" and optimizer is not None:
                        f = float(patch["factor"])
                        for i, g in enumerate(optimizer.param_groups):
                            g["lr"] *= f
                            if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
                                lr_scheduler.base_lrs[i] *= f
                    elif patch.get("kind") == "grad_clip":
                        args.max_grad_norm = float(patch["max_grad_norm"])
                    elif patch.get("kind") == "eval_every":  # чекпоинты — на тех же шагах, что и оценки
                        state.eval_steps = state.save_steps = int(patch["steps"])
                    elif patch.get("kind") == "log_every":
                        state.logging_steps = int(patch["steps"])
                append_jsonl(paths.events, {"type": "control_applied", "cmd": kind, "id": cmd.get("id"),
                                            "step": step_offset + state.global_step})

    return Recorder(), Control()


def _mlflow_log(recorder, values: dict, step: int):
    if recorder.mlflow is None:
        return
    try:
        recorder.mlflow.log_metrics({k: float(v) for k, v in values.items() if _finite(v)}, step=step)
    except Exception:
        recorder.mlflow = None


def _start_mlflow(paths: RunPaths, round_k: int, cfg: RunConfig):
    try:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # 3.17: файловое хранилище только по явному согласию
        import mlflow
        app_cfg = get_config()
        mlflow.set_tracking_uri((app_cfg.data_dir / app_cfg.mlflow_tracking_dir).resolve().as_uri())
        mlflow.set_experiment("lora-factory")
        mlflow.start_run(run_name=f"{paths.run_id}/round_{round_k}")
        flat = {"backend": cfg.backend, "model": cfg.base_model_id, "lr": cfg.optim.lr, "epochs": cfg.schedule.epochs,
                "micro_batch": cfg.schedule.micro_batch, "grad_accum": cfg.schedule.grad_accum,
                "max_seq_len": cfg.max_seq_len, "quantization": cfg.quantization}
        if cfg.lora:
            flat.update(lora_r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_variant=cfg.lora.variant)
        if cfg.ptuning:
            flat.update(num_virtual_tokens=cfg.ptuning.num_virtual_tokens)
        mlflow.log_params(flat)
        return mlflow
    except Exception as e:
        logger.warning("MLflow недоступен: %s", e)
        return None


# ---------------------------------------------------------------- тренер

def make_trainer(model, tok, cfg: RunConfig, train_ds, eval_ds, out_dir: Path, evaluator, recorder, control_cb,
                 pilot_steps: int | None, epochs: float, lr: float, scheduler: str, warmup: float,
                 resume_optimizer: Path | None, max_steps: int | None = None):
    import torch
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    best = cfg.schedule.best_metric
    greater = best != "eval_loss"
    is_pilot = pilot_steps is not None
    dtype = _dtype()
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        max_steps=pilot_steps if is_pilot else (max_steps or -1),
        per_device_train_batch_size=cfg.schedule.micro_batch,
        per_device_eval_batch_size=cfg.schedule.micro_batch,
        gradient_accumulation_steps=cfg.schedule.grad_accum,
        learning_rate=lr,
        lr_scheduler_type=scheduler,
        lr_scheduler_kwargs={"min_lr_rate": 0.1} if scheduler == "cosine_with_min_lr" else None,
        warmup_steps=warmup,
        weight_decay=cfg.optim.weight_decay,
        max_grad_norm=cfg.optim.max_grad_norm,
        optim=cfg.optim.optimizer,
        logging_steps=1 if is_pilot else cfg.schedule.logging_steps,
        logging_first_step=True,
        eval_strategy="no" if is_pilot else "steps",
        eval_steps=cfg.schedule.eval_steps,
        save_strategy="no" if is_pilot else "steps",
        save_steps=cfg.schedule.eval_steps,
        save_total_limit=2,
        save_only_model=False,
        load_best_model_at_end=not is_pilot,
        metric_for_best_model=best,
        greater_is_better=greater,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=cfg.schedule.gradient_checkpointing and cfg.backend != "unsloth_lora",
        max_length=cfg.max_seq_len,
        packing=False,
        report_to="none",
        seed=cfg.seeds.train_seed,
        data_seed=cfg.seeds.train_seed,
        disable_tqdm=True,
        dataloader_num_workers=0,
    )

    class FactoryTrainer(SFTTrainer):
        def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                            metric_key_prefix="eval"):
            from factory.training.evaluator import memory_note
            logger.info("Память перед оценкой: %s", memory_note())
            out = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
            logger.info("Память после val loss: %s", memory_note())
            res = evaluator.run(self.model, tok, cheap_only=True)
            logger.info("Память после генерации: %s", memory_note())
            for k, v in res["metrics"].items():
                out.metrics[f"{metric_key_prefix}_{k}"] = v
            recorder.record_eval(self.state, out.metrics, res["pred_label_share"])
            return out

        def create_optimizer(self, *a, **kw):
            opt = super().create_optimizer(*a, **kw)
            if resume_optimizer is not None and resume_optimizer.exists():
                try:
                    opt.load_state_dict(torch.load(resume_optimizer, map_location="cpu", weights_only=False))
                    logger.info("Состояние оптимизатора восстановлено из %s", resume_optimizer)
                except Exception as e:
                    logger.warning("Состояние оптимизатора не загружено (%s) — новый оптимизатор", e)
                for g in opt.param_groups:  # lr продолжения: не выше, чем в конце прошлого раунда
                    g["lr"] = lr
                    g["initial_lr"] = lr
            return opt

    callbacks = [recorder, control_cb]
    if not is_pilot:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.schedule.patience))
    return FactoryTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                          processing_class=tok, callbacks=callbacks)


# ---------------------------------------------------------------- режимы

class Ctx:
    def __init__(self, run_dir: Path):
        from factory.training.formatting import instruction
        self.paths = RunPaths(run_dir)
        self.manifest = read_model(self.paths.manifest, Manifest)
        self.app = get_config()
        registry = Registry.load()
        self.task = registry.task(self.manifest.spec.task_type)
        model = registry.models.get(self.manifest.spec.base_model_id)
        self.chat = bool(model and model.chat_template)
        self.max_context = model.max_context if model else None
        self.instr = instruction(self.manifest.spec, self.task)
        self.recipe = read_model(self.paths.recipe, Recipe)
        self.status_file: Path | None = None

    def progress(self, text: str) -> None:
        """Ход долгой операции (генерация сотен ответов идёт десятки минут): в файл статуса — его показывает лента
        «что происходит», и в heartbeat — иначе супервизор сочтёт процесс зависшим."""
        if self.status_file:
            write_json(self.status_file, {"status": "running", "pid": os.getpid(), "progress": text})
        self.paths.heartbeat.write_text(str(time.time()))

    def eval_frame(self, df):
        """val/test с теми же переименованиями и фильтрами меток, что и train."""
        from factory.prep.ops import apply_label_schema
        return apply_label_schema(df, self.recipe) if self.recipe else df

    def prompt(self, tok):
        from factory.training.formatting import prompt_builder
        return prompt_builder(self.instr, tok, self.chat)

    def config_for(self, round_dir: Path | None) -> RunConfig:
        if round_dir is not None and (round_dir / "config.json").exists():
            return read_model(round_dir / "config.json", RunConfig)
        return read_model(self.paths.config, RunConfig)

    def frames(self):
        import pandas as pd
        return (pd.read_parquet(self.paths.data / "train.parquet"),
                self.eval_frame(pd.read_parquet(self.paths.data / "val.parquet")))


def compute_baselines(ctx: Ctx, evaluator, train_df, model, tok) -> dict[str, dict[str, float]]:
    """Baseline'ы из реестра задачи: простые (без модели) и исходная модель с тем же промптом (адаптер выключен)."""
    from factory.training import metrics as m
    targets = train_df["target"].tolist()
    out = {}
    for name in ctx.task.baselines:
        if name == "base_prompt":
            evaluator.on_progress = lambda done, total: ctx.progress(f"ответы исходной модели (baseline): {done} из {total}")
            if hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    preds = evaluator.predict(model, tok)
            else:
                preds = evaluator.predict(model, tok)
        else:
            preds = m.simple_baseline_predictions(name, evaluator.original_inputs, targets, [len(t) for t in targets[:2000]])
        if preds is not None:
            ctx.progress(f"метрики baseline {name}")
            out[name] = evaluator.score_predictions(preds)["metrics"]
    return out


def mode_baseline(ctx: Ctx) -> dict:
    cfg = ctx.config_for(None)
    train_df, val_df = ctx.frames()
    model, tok = load_base_model(cfg)
    evaluator = make_evaluator(ctx, val_df, tok, ctx.app.training.num_eval_samples, cfg)
    baselines = compute_baselines(ctx, evaluator, train_df, model, tok)
    write_json(ctx.paths.baselines_val, baselines)
    return {"status": "ok", "baselines": baselines}


def mode_pilot(ctx: Ctx, steps: int) -> dict:
    import torch
    cfg = ctx.config_for(ctx.paths.pilot)
    train_df, val_df = ctx.frames()
    model, tok = build_model(cfg)
    n_rows = min(len(train_df), steps * cfg.schedule.micro_batch * cfg.schedule.grad_accum)
    train_ds = sft_dataset(train_df.sample(n=n_rows, random_state=cfg.seeds.train_seed), tok, ctx.prompt(tok), cfg.max_seq_len)
    tps_step = tokens_per_step(train_ds, tok, cfg)
    recorder, control_cb = make_callbacks(ctx.paths, 0, 0, tps_step, pilot=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = make_trainer(model, tok, cfg, train_ds, None, ctx.paths.pilot / "out", None, recorder, control_cb,
                           pilot_steps=steps, epochs=1, lr=cfg.optim.lr, scheduler=cfg.optim.scheduler,
                           warmup=cfg.optim.warmup, resume_optimizer=None)
    trainer.train()
    if recorder.nan:
        return {"status": "nan", "message": "loss или норма градиента стали NaN/inf в пилоте"}
    times = recorder.step_times[3:] if len(recorder.step_times) > 6 else recorder.step_times
    sec_per_step = (times[-1] - times[0]) / max(len(times) - 1, 1) if len(times) > 1 else float("nan")
    evaluator = make_evaluator(ctx, val_df, tok, 16, cfg)
    eval_probe = evaluator.run(trainer.model, tok)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None
    from factory.hparams.resources import steps_per_epoch
    spe = steps_per_epoch(len(train_df), cfg.schedule.micro_batch, cfg.schedule.grad_accum)
    total_steps = spe * cfg.schedule.epochs
    n_evals = total_steps // cfg.schedule.eval_steps + 1
    eval_s = eval_probe["elapsed_s"] / 16 * min(len(val_df), BEST_EVAL_SAMPLES_DURING_TRAINING) * n_evals
    est = total_steps * sec_per_step * 1.15 + eval_s
    return {
        "status": "ok", "steps": steps, "sec_per_step": round(sec_per_step, 4),
        "tokens_per_sec": round(tps_step / sec_per_step, 1) if sec_per_step == sec_per_step and sec_per_step > 0 else None,
        "peak_mem_gb": round(peak, 3) if peak else None,
        "total_steps": total_steps, "steps_per_epoch": spe, "estimated_time_s": round(est, 1),
        "final_loss": next((l["loss"] for l in reversed(trainer.state.log_history) if "loss" in l), None),
    }


def _last_checkpoint(round_dir: Path) -> Path | None:
    from factory.training.checkpoints import last_checkpoint
    return last_checkpoint(round_dir)


def mode_train(ctx: Ctx, round_k: int, epochs: float, resume_from: Path | None, lr_override: float | None,
               step_offset: int, epoch_offset: float) -> dict:
    rdir = ctx.paths.round_dir(round_k)
    rdir.mkdir(parents=True, exist_ok=True)
    from factory.training.checkpoints import new_attempt
    ctx.checkpoint_dir = new_attempt(rdir)
    cfg = ctx.config_for(rdir)
    train_df, val_df = ctx.frames()
    model, tok = build_model(cfg, resume_adapter=resume_from)
    train_ds = sft_dataset(train_df, tok, ctx.prompt(tok), cfg.max_seq_len)
    eval_ds = sft_dataset(val_df.sample(n=min(len(val_df), 512), random_state=cfg.seeds.split_seed), tok, ctx.prompt(tok), cfg.max_seq_len)
    evaluator = make_evaluator(ctx, val_df, tok, BEST_EVAL_SAMPLES_DURING_TRAINING, cfg)
    recorder, control_cb = make_callbacks(ctx.paths, step_offset, epoch_offset, tokens_per_step(train_ds, tok, cfg), pilot=False)
    recorder.mlflow = _start_mlflow(ctx.paths, round_k, cfg)

    continuing = resume_from is not None
    lr = lr_override if lr_override is not None else cfg.optim.lr
    trainer = make_trainer(
        model, tok, cfg, train_ds, eval_ds, ctx.checkpoint_dir, evaluator, recorder, control_cb, pilot_steps=None, epochs=epochs,
        lr=lr, scheduler="constant" if continuing else cfg.optim.scheduler,
        warmup=0.0 if continuing else cfg.optim.warmup,
        resume_optimizer=(resume_from / "optimizer.pt") if continuing else None,
        # бюджет GPU-часов меньше эпохи — предел шагов первого раунда; продолжение пользователь заказывает сам
        max_steps=None if continuing else cfg.schedule.max_steps)
    trainer.train()
    if recorder.nan:
        return {"status": "nan", "message": "loss или норма градиента стали NaN/inf",
                "checkpoint": str(_last_checkpoint(ctx.checkpoint_dir) or ""), "global_step": step_offset + trainer.state.global_step}
    if control_cb.aborted:
        return {"status": "aborted", "global_step": step_offset + trainer.state.global_step}

    best_dir = rdir / "adapter_best"
    trainer.model.save_pretrained(str(best_dir))  # load_best_model_at_end → в памяти лучший адаптер
    tok.save_pretrained(str(best_dir))
    ckpt = _last_checkpoint(ctx.checkpoint_dir)
    lr_end = next((l["learning_rate"] for l in reversed(trainer.state.log_history) if "learning_rate" in l), lr)

    full_eval = make_evaluator(ctx, val_df, tok, ctx.app.training.num_eval_samples, cfg)
    res = full_eval.run(trainer.model, tok, cheap_only=False)
    baselines = read_json(ctx.paths.baselines_val)
    if baselines is None:
        baselines = compute_baselines(ctx, full_eval, train_df, trainer.model, tok)
        write_json(ctx.paths.baselines_val, baselines)
    if recorder.mlflow is not None:
        try:
            recorder.mlflow.log_metrics({f"round_val_{k}": v for k, v in res["metrics"].items()})
            recorder.mlflow.end_run()
        except Exception:
            pass
    return {
        "status": "ok",
        "global_step": step_offset + trainer.state.global_step,
        "epochs_done": round(epoch_offset + (trainer.state.epoch or 0), 4),
        "checkpoint": str(ckpt) if ckpt else None,
        "adapter_best": str(best_dir),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "lr_end": lr_end,
        "val_metrics": res["metrics"],
        "unparsed_fraction": res["unparsed_fraction"],
        "baselines_val": baselines,
        "early_stopped": trainer.state.global_step < trainer.state.max_steps,
    }


def mode_final_eval(ctx: Ctx, adapter: Path, round_k: int | None = None) -> dict:
    """Окончательная остановка: test открывается один раз."""
    from peft import PeftModel

    from factory.prep.split import read_test
    from factory.training import metrics as m
    cfg = ctx.config_for(ctx.paths.round_dir(round_k) if round_k else None)
    if cfg.backend == "unsloth_lora":
        model, tok = load_unsloth(str(adapter), cfg)  # Unsloth сам подтянет базу из adapter_config
    else:
        base, tok = load_base_model(cfg)
        model = PeftModel.from_pretrained(base, str(adapter))
    train_df, _ = ctx.frames()
    test_df = ctx.eval_frame(read_test(ctx.paths.data, final=True, expected_hash=ctx.manifest.test_hash,
                                       events_path=ctx.paths.events))
    evaluator = make_evaluator(ctx, test_df, tok, 2000, cfg, "ответы адаптера", ctx.app.training.final_eval_token_budget)
    preds = evaluator.predict(model, tok)
    ctx.progress("метрики адаптера")
    res = evaluator.score_predictions(preds)
    baselines = compute_baselines(ctx, evaluator, train_df, model, tok)
    alts = evaluator.alternatives or [None] * len(preds)
    correct = [m.example_correct(ctx.task.format, p, n, r, a)
               for p, n, r, a in zip(preds, res["normalized_predictions"], evaluator.references, alts)]
    ok, bad = m.examples(evaluator.original_inputs, evaluator.references, preds, correct, get_config().training.test_examples_shown)
    report = {
        "split": "test", "n_examples": len(preds), "metrics": res["metrics"], "per_class": res["per_class"],
        "latency_ms": round(evaluator.latency_ms(model, tok), 1),
        "baselines": baselines, "unparsed_fraction": res["unparsed_fraction"],
        "examples_correct": [e.model_dump() for e in ok], "examples_wrong": [e.model_dump() for e in bad],
    }
    write_json(ctx.paths.eval_report, report)
    return {"status": "ok", "report": str(ctx.paths.eval_report)}


# ---------------------------------------------------------------- точка входа

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--mode", required=True, choices=["baseline", "pilot", "train", "final_eval"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--resume-from", type=Path, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--step-offset", type=int, default=0)
    ap.add_argument("--epoch-offset", type=float, default=0.0)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--status-file", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    write_json(a.status_file, {"status": "queued", "pid": os.getpid()})
    ctx = Ctx(a.run_dir)
    ctx.status_file = a.status_file
    round_dir = ctx.paths.pilot if a.mode == "pilot" else ctx.paths.round_dir(a.round)
    if ctx.config_for(round_dir).backend == "unsloth_lora":
        import unsloth  # noqa: F401 — патчит transformers/trl/peft, поэтому импортируется первым
    with file_lock(ctx.app.locks_dir / f"gpu{a.gpu}.lock"):  # очередь GPU: один процесс на устройство
        write_json(a.status_file, {"status": "running", "pid": os.getpid()})
        from factory.training.evaluator import cap_vram
        cap = cap_vram()  # после очереди: прошлый процесс уже отдал память
        if cap is not None:
            logger.info("Предел видеопамяти процесса: %.1f ГБ (остальное занято другими программами или в запасе)", cap)
        modes = {
            "baseline": lambda: mode_baseline(ctx),
            "pilot": lambda: mode_pilot(ctx, a.steps),
            "train": lambda: mode_train(ctx, a.round, a.epochs or ctx.config_for(ctx.paths.round_dir(a.round)).schedule.epochs,
                                        a.resume_from, a.lr, a.step_offset, a.epoch_offset),
            "final_eval": lambda: mode_final_eval(ctx, a.adapter, a.round),
        }
        try:
            result = modes[a.mode]()
        except Exception as e:
            import torch
            if isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower():
                result = {"status": "oom", "message": str(e)[:500], "traceback": traceback.format_exc()[-4000:]}
                logger.error("Нехватка памяти GPU:\n%s", traceback.format_exc())
                # супервизор продолжит с чекпоинта этого процесса, а не начнёт раунд заново. Только своего: номер
                # чекпоинта — шаг внутри процесса, и чекпоинт прошлого процесса раунда со смещением шагов этого
                # процесса дал бы двойной счёт (без своего супервизор повторит старт этого процесса — верно)
                if a.mode == "train" and getattr(ctx, "checkpoint_dir", None):
                    own = _last_checkpoint(ctx.checkpoint_dir)
                    if own:
                        result["checkpoint"] = str(own)
            else:
                result = {"status": "error", "message": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
            logger.error("Рабочий процесс: %s", result.get("message"))
    write_json(a.status_file, result | {"pid": os.getpid(), "finished_at": time.time()})
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
