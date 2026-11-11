"""Агенты без сети: FunctionModel воспроизводит ответы LLM, проверяем инструменты, валидаторы и guard'ы."""
import asyncio
import json

import pandas as pd
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from factory.core import activity, llm


def scripted(*steps):
    """FunctionModel, отдающий заранее заданную последовательность ответов (вызовы инструментов / финальный ответ)."""
    calls = iter(steps)

    def fn(messages, info: AgentInfo):
        step = next(calls)
        if callable(step):
            step = step(messages, info)
        return step
    return FunctionModel(fn)


def tool(name, **args):
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])


def final(info: AgentInfo, out_name: str | None = None, **args):
    names = [t.name for t in info.output_tools]
    name = out_name if out_name in names else next(n for n in names if out_name is None or out_name.lower() in n.lower())
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ A1

def test_intake_ask_then_spec(registry):
    from factory.agents import intake
    deps = intake.IntakeDeps(registry=registry)
    model = scripted(
        lambda m, i: final(i, "AskUser", question="Какие метки?", options=["позитив/негатив"]),
    )
    with llm.override_model(model):
        out, conv = run(intake.run_turn(deps, "хочу классифицировать отзывы"))
    assert isinstance(out, intake.AskUser) and out.options

    bad_model = {"description": "Тональность отзывов", "task_type": "sentiment_analysis", "language": "ru",
                 "labels": ["позитив", "негатив"], "base_model_id": "gpt-5"}
    good = dict(bad_model, base_model_id="Qwen/Qwen3-0.6B")
    model = scripted(
        lambda m, i: tool("list_models"),
        lambda m, i: final(i, "ProposeSpec", spec=bad_model),   # модель не из списка → ModelRetry
        lambda m, i: final(i, "ProposeSpec", spec=good),
    )
    with llm.override_model(model):
        out, conv = run(intake.run_turn(deps, "позитив и негатив", conv))
    assert isinstance(out, intake.ProposeSpec)
    assert deps.finalized_spec.primary_metric == "macro_f1"  # из реестра, не от LLM


def test_intake_present_only_found(registry, cls_spec):
    from factory.agents import intake
    from factory.core.contracts import DatasetCandidate

    async def fake_search(spec, extra):
        return [DatasetCandidate(hf_id="org/reviews", score=80, why="подходит")]

    deps = intake.IntakeDeps(registry=registry, spec=cls_spec.model_copy(update={"confirmed_by_user": True}), search=fake_search)
    model = scripted(
        lambda m, i: tool("search_datasets"),
        lambda m, i: final(i, "PresentDatasets", hf_ids=["org/invented"], message="вот"),
        lambda m, i: final(i, "PresentDatasets", hf_ids=["org/reviews"], message="вот"),
    )
    with llm.override_model(model):
        out, _ = run(intake.run_turn(deps, "найди"))
    assert out.hf_ids == ["org/reviews"]


def test_search_before_reconfirming_spec_does_not_exhaust_retries(registry, cls_spec):
    from factory.agents import intake

    async def unexpected_search(spec, extra):
        raise AssertionError("unconfirmed spec must not start a dataset search")

    deps = intake.IntakeDeps(registry=registry,
                             spec=cls_spec.model_copy(update={"confirmed_by_user": False}),
                             search=unexpected_search)
    revised = {"description": "Классификация англоязычных отзывов", "task_type": "sentiment_analysis",
               "language": "en", "labels": ["positive", "negative"], "base_model_id": "Qwen/Qwen3-0.6B"}
    model = scripted(*(lambda m, i: tool("search_datasets") for _ in range(4)),
                     lambda m, i: final(i, "ProposeSpec", spec=revised))
    with llm.override_model(model):
        out, _ = run(intake.run_turn(deps, "Ослабить ограничения: можно английский язык"))
    assert isinstance(out, intake.ProposeSpec)
    assert deps.finalized_spec.language == "en"


def test_conversation_roundtrip(registry):
    from pydantic_ai import ConversationTypeAdapter

    from factory.agents import intake
    deps = intake.IntakeDeps(registry=registry)
    with llm.override_model(scripted(lambda m, i: final(i, "AskUser", question="?"))):
        _, conv = run(intake.run_turn(deps, "привет"))
    restored = ConversationTypeAdapter.validate_json(ConversationTypeAdapter.dump_json(conv))
    with llm.override_model(scripted(lambda m, i: final(i, "AskUser", question="ещё?"))):
        out, conv2 = run(intake.run_turn(deps, "ещё", restored))
    assert len(conv2.messages) > len(conv.messages)


# ------------------------------------------------------------------ A2

def _workspace(cfg, tmp_path, mode="auto"):
    from factory.agents.dataprep import PrepWorkspace
    rows = [(f"хороший отличный товар номер {i}, всем рекомендую", "pos") for i in range(50)]
    rows += [(f"плохой ужасный товар номер {i}, сломался через день", "neg") for i in range(5)]
    df = pd.DataFrame(rows, columns=["input", "target"])
    df["synthetic"] = False
    df.insert(0, "_id", [str(i) for i in range(len(df))])
    return PrepWorkspace(sample=df, family="classification", language="ru", labels=["pos", "neg"],
                         task_description="тональность", cfg=cfg.dataprep, mode=mode, synth_dir=tmp_path, seed=0,
                         findings=["дисбаланс"], full_train_rows=len(df))


def test_dataprep_agent_builds_recipe(cfg, tmp_path):
    from factory.agents import dataprep
    ws = _workspace(cfg, tmp_path)
    model = scripted(
        lambda m, i: tool("profile"),
        lambda m, i: tool("rebalance", method="undersample", max_ratio=3.0, apply=False),
        lambda m, i: tool("rebalance", method="undersample", max_ratio=3.0, apply=True),
        lambda m, i: final(i, rationale="поравнял классы"),  # без commit_recipe → ModelRetry
        lambda m, i: tool("commit_recipe", rationale="undersample до 3:1"),
        lambda m, i: final(i, rationale="поравнял классы"),
    )
    with llm.override_model(model):
        run(dataprep.run(ws))
    assert ws.committed and [o.op for o in ws.ops] == ["rebalance"]
    vc = ws.current["target"].value_counts()
    assert vc.max() / vc.min() <= 3.0


def test_dataprep_label_noise_capped_by_config(cfg, tmp_path):
    from factory.agents import dataprep
    ws = _workspace(cfg, tmp_path)
    model = scripted(
        lambda m, i: tool("drop_label_noise", max_fraction=0.4, apply=True),  # больше лимита → ModelRetry
        lambda m, i: tool("commit_recipe", rationale="ничего"),
        lambda m, i: final(i, rationale="ок"),
    )
    with llm.override_model(model):
        run(dataprep.run(ws))
    assert not any(o.op == "drop_label_noise" for o in ws.ops)


def test_dataprep_augment_needs_approval_in_expert(cfg, tmp_path):
    from factory.agents import dataprep
    ws = _workspace(cfg, tmp_path, mode="expert")
    model = scripted(
        lambda m, i: tool("augment", method="llm", n_rows=100, confidence=0.99, reason="редкий класс"),
        lambda m, i: tool("commit_recipe", rationale="предложен синтез"),
        lambda m, i: final(i, rationale="ок"),
    )
    with llm.override_model(model):
        run(dataprep.run(ws))
    assert ws.augment_proposal and ws.augment_proposal["n_rows"] == 100
    assert not any(o.op == "augment" for o in ws.ops)


def test_dataprep_augment_auto_when_confident(cfg, tmp_path, monkeypatch):
    from factory.agents import dataprep
    from factory.prep import augment

    async def synonyms(texts, language, n_words=150):
        return {"плохой": ["скверный"], "ужасный": ["кошмарный"]}
    monkeypatch.setattr(augment, "build_synonyms", synonyms)
    ws = _workspace(cfg, tmp_path, mode="auto")
    model = scripted(
        lambda m, i: tool("augment", method="deterministic", n_rows=30, confidence=0.95, reason="редкий класс"),
        lambda m, i: tool("commit_recipe", rationale="синтез"),
        lambda m, i: final(i, rationale="ок"),
    )
    with llm.override_model(model):
        run(dataprep.run(ws))
    aug = [o for o in ws.ops if o.op == "augment"]
    assert aug and aug[0].n_rows > 0
    assert ws.current["synthetic"].sum() == aug[0].n_rows
    synth = ws.current[ws.current["synthetic"]]["input"]
    assert synth.str.contains("скверный|кошмарный").any()  # синонимизация по словарю


def test_dataprep_augment_shows_train_share_in_sample(cfg, tmp_path, monkeypatch):
    """Синтетика идёт во весь train: в выборке — только её доля, иначе дисбаланс «исправился» бы лишь на выборке."""
    from factory.agents import dataprep
    from factory.prep import augment

    async def synonyms(texts, language, n_words=150):
        return {"плохой": ["скверный"], "ужасный": ["кошмарный"]}
    monkeypatch.setattr(augment, "build_synonyms", synonyms)
    ws = _workspace(cfg, tmp_path, mode="auto")
    ws.full_train_rows = len(ws.sample) * 10
    model = scripted(
        lambda m, i: tool("augment", method="deterministic", n_rows=30, confidence=0.95, reason="редкий класс"),
        lambda m, i: tool("commit_recipe", rationale="синтез", waivers={"class_balance": "дисбаланс отражает реальные данные"}),
        lambda m, i: final(i, rationale="ок"),
    )
    with llm.override_model(model):
        run(dataprep.run(ws))
    aug = next(o for o in ws.ops if o.op == "augment")
    assert ws.current["synthetic"].sum() == round(aug.n_rows / 10)


# ------------------------------------------------------------------ A3

def test_hparam_agent_requires_real_evidence(cfg, registry, cls_spec, tmp_path):
    from factory.agents import hparam
    from factory.core.contracts import DataProfile, LengthStats, Seeds
    from factory.hparams.guard import HParamContext
    from factory.runs import RunPaths

    ctx = HParamContext(spec=cls_spec, task=registry.task(cls_spec.task_type), model=registry.model("Qwen/Qwen3-0.6B"),
                        profile=DataProfile(n_rows=1000, input_tokens_est=LengthStats(p99=100), target_tokens_est=LengthStats(p99=3)),
                        n_train=1000, vram_gb=12, cfg=cfg, seeds=Seeds(split_seed=1, train_seed=1))
    deps = hparam.HParamDeps(ctx=ctx, paths=RunPaths(tmp_path), baselines_val={})
    chunk = {}

    def answer(doc_ref):
        return lambda m, i: final(i, lr=2e-4, epochs=3, lora={"r": 8, "alpha": 16}, rationales=[
            {"param": "lr", "value": 2e-4, "reason": "заметки", "evidence": [{"kind": "doc", "ref": doc_ref}]},
            {"param": "epochs", "value": 3, "reason": "заметки", "evidence": [{"kind": "doc", "ref": doc_ref}]},
            {"param": "lora.r", "value": 8, "reason": "заметки", "evidence": [{"kind": "doc", "ref": doc_ref}]},
        ])

    def lookup_then_answer(m, i):
        # берём реальный doc_chunk_id из результата инструмента
        for msg in reversed(m):
            for part in getattr(msg, "parts", []):
                if getattr(part, "tool_name", None) == "lookup_hparam_docs" and hasattr(part, "content"):
                    chunk["id"] = part.content[0]["doc_chunk_id"]
        return answer(chunk["id"])(m, i)

    model = scripted(
        lambda m, i: tool("find_similar_runs"),
        lambda m, i: tool("lookup_hparam_docs", query="lora learning rate"),
        answer("выдуманный.md#0"),  # несуществующее свидетельство → ModelRetry
        lookup_then_answer,
    )
    with llm.override_model(model):
        out = run(hparam.run(deps))
    assert out.lr == 2e-4 and chunk["id"] in deps.seen_chunks


def test_strategy_falls_back_to_heuristic(cfg, registry, cls_spec, tmp_path, monkeypatch):
    from factory.core.contracts import DataProfile, LengthStats, Seeds
    from factory.hparams import strategy
    from factory.hparams.guard import HParamContext
    from factory.hparams.pilot import PilotReport
    from factory.runs import RunPaths

    async def fake_pilot(paths, config, n_train, pc):
        return PilotReport(status="ok", config=config, result={"status": "ok"})
    monkeypatch.setattr(strategy, "run_pilot", fake_pilot)

    def broken(m, i):
        raise RuntimeError("LLM упала")
    ctx = HParamContext(spec=cls_spec, task=registry.task(cls_spec.task_type), model=registry.model("Qwen/Qwen3-0.6B"),
                        profile=DataProfile(n_rows=1000, input_tokens_est=LengthStats(p99=100), target_tokens_est=LengthStats(p99=3)),
                        n_train=1000, vram_gb=12, cfg=cfg, seeds=Seeds(split_seed=1, train_seed=1))
    with llm.override_model(FunctionModel(broken)):
        p = run(strategy.propose(ctx, RunPaths(tmp_path), {}, "agent"))
    assert p.strategy == "heuristic" and p.fallback_reason and p.config.strategy == "heuristic"


# ------------------------------------------------------------------ A4

class FakeLive:
    """Живой процесс для A4 без обучения: записывает отправленные команды."""

    def __init__(self):
        self.sent = []

    def recent_metrics(self, n):
        return [{"step": 100, "train_loss": 1.2, "lr": 2e-4}, {"step": 110, "train_loss": 3.9, "lr": 2e-4}]

    def val_curve(self):
        return [{"step": 100, "val_loss": 1.1, "macro_f1": 0.6}]

    def config_summary(self):
        return {"lr": 2e-4, "eval_steps": 50}

    def event_history(self, n):
        return []

    async def wait_new(self, seconds):
        return [{"step": 120, "train_loss": 1.0, "lr": 1e-4}]

    async def send(self, cmd, patch, decision):
        self.sent.append((cmd, patch, decision.action))
        return {"applied": True, "at_step": 111, "command": cmd}


def test_monitor_controls_live_process_within_limits():
    from factory.agents import monitor
    live = FakeLive()
    deps = monitor.MonitorDeps(task_type="t", primary_metric="macro_f1", allowed_actions=["continue", "patch"],
                               interventions_left=1, live=live, eval_steps=50)
    model = scripted(
        lambda m, i: tool("get_recent_metrics", n=5),
        lambda m, i: tool("stop_training", reason="loss скачет"),          # не разрешено настройками → ModelRetry
        lambda m, i: tool("scale_lr", factor=1.5, reason="ускорить"),      # только уменьшение → ModelRetry
        lambda m, i: tool("scale_lr", factor=0.5, reason="скачок loss"),
        lambda m, i: tool("scale_lr", factor=0.5, reason="ещё раз"),       # лимит исчерпан → ModelRetry
        lambda m, i: tool("set_eval_every", steps=20, reason="следить плотнее"),  # наблюдение — не в лимит
        lambda m, i: tool("request_eval_now", reason="проверить после поправки"),
        lambda m, i: final(i, verdict="suspicious", explanation="Снизил lr вдвое, оценка чаще."),
    )
    with llm.override_model(model):
        v = run(monitor.review(deps, "аномалии детекторов", [{"kind": "nan_loss", "severity": "soft"}]))
    assert v.verdict == "suspicious"
    assert live.sent == [("patch", {"kind": "lr_scale", "factor": 0.5}, "patch"),
                         ("patch", {"kind": "eval_every", "steps": 20}, "patch"),
                         ("eval_now", None, "eval_now")]
    assert deps.interventions_left == 0 and len(deps.actions) == 3


def test_monitor_cannot_finish_when_code_says_insufficient():
    from factory.agents import monitor
    from factory.training.rounds import flatten_values
    values = flatten_values(val={"macro_f1": 0.4}, baseline={"majority": {"macro_f1": 0.5}})
    deps = monitor.MonitorDeps(task_type="t", primary_metric="macro_f1", values=values, code_sufficient=False)
    model = scripted(
        lambda m, i: final(i, diagnosis="Всё хорошо", recommendation="finish"),
        lambda m, i: final(i, diagnosis="F1 0.40 ниже", recommendation="continue", extra_epochs=2),  # цифры без плейсхолдеров
        lambda m, i: final(i, diagnosis="F1 {val.macro_f1} ниже baseline {baseline.majority.macro_f1}",
                           recommendation="continue", extra_epochs=2),
    )
    with llm.override_model(model):
        a = run(monitor.assess_round(deps, 1, 2.0, "ниже baseline", [], False))
    assert a.recommendation == "continue" and "{val.macro_f1}" in a.diagnosis


def test_llm_calls_are_logged(tmp_path):
    from factory.agents import monitor
    log = tmp_path / "llm_calls.jsonl"
    deps = monitor.MonitorDeps(task_type="t", primary_metric="macro_f1", live=FakeLive())
    with llm.override_model(scripted(lambda m, i: final(i, verdict="ok", explanation="ок"))), llm.llm_log_to(log):
        run(monitor.review(deps, "плановая проверка", []))
    rec = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert rec["role"] == "monitor" and rec["ok"] and rec["prompt_hash"]


def test_transient_error_resumes_without_repeating_actions(monkeypatch):
    """Сетевая ошибка посреди прогона: повтор продолжает с места сбоя — команда A4 не уходит процессу дважды."""
    from pydantic_ai.exceptions import ModelAPIError

    from factory.agents import monitor
    sleep = asyncio.sleep
    monkeypatch.setattr(llm.asyncio, "sleep", lambda s: sleep(0))
    failed = []

    def model(messages, info):  # как настоящая модель: решает по истории — команды ещё не было → отдать её
        acted = any(p.part_kind == "tool-return" for msg in messages for p in msg.parts)
        if not acted:
            return tool("set_grad_clip", max_norm=0.5, reason="взрыв градиента")
        if not failed:
            failed.append(1)
            raise ModelAPIError("m", "Connection error.")
        return final(info, verdict="suspicious", explanation="Ограничил градиент.")
    live = FakeLive()
    deps = monitor.MonitorDeps(task_type="t", primary_metric="macro_f1", live=live, eval_steps=50)
    with llm.override_model(FunctionModel(model)):
        v = run(monitor.review(deps, "аномалии", []))
    assert failed and v.explanation == "Ограничил градиент."
    assert [s[1] for s in live.sent] == [{"kind": "grad_clip", "max_grad_norm": 0.5}] and deps.interventions_left == 2


def _dirty_workspace(cfg, tmp_path):
    import random

    from factory.agents.dataprep import PrepWorkspace
    rng = random.Random(0)
    words = ["отличный", "сервис", "быстро", "вкусно", "уютно", "дорого", "медленно", "грязно", "шумно", "приятно"]
    rows = [(" ".join(rng.sample(words, 6)) + f" отзыв {i}", "pos" if i % 2 else "neg") for i in range(120)]
    rows += [(f"<p>{' '.join(rng.sample(words, 5))}</p> &nbsp; заметка {i}", "pos") for i in range(12)]  # HTML-мусор
    df = pd.DataFrame(rows, columns=["input", "target"])
    df["synthetic"] = False
    df.insert(0, "_id", [str(i) for i in range(len(df))])
    return PrepWorkspace(sample=df, family="classification", language="ru", labels=["pos", "neg"],
                         task_description="тональность", cfg=cfg.dataprep, mode="expert", synth_dir=tmp_path, seed=0,
                         findings=[], full_train_rows=len(df))


def test_dataprep_cannot_commit_until_goals_met(cfg, tmp_path, monkeypatch):
    from factory.agents import dataprep
    from factory.prep import quality
    monkeypatch.setattr(quality.lang, "language_share", lambda texts: {"ru": 1.0})  # язык не проверяем в этом тесте
    ws = _dirty_workspace(cfg, tmp_path)
    assert "junk_text" in quality.failing(ws.report())
    model = scripted(
        lambda m, i: tool("check_quality"),
        lambda m, i: tool("commit_recipe", rationale="всё ок"),                       # мусор не убран → ModelRetry
        lambda m, i: tool("commit_recipe", rationale="ок", waivers={"junk_text": "лень"}),  # отписка < 15 символов
        lambda m, i: tool("junk_examples"),
        lambda m, i: tool("clean_text", apply=True),
        lambda m, i: tool("check_quality"),
        lambda m, i: tool("commit_recipe", rationale="почистил HTML"),
        lambda m, i: final(i, rationale="Удалена HTML-разметка, строки сохранены."),
    )
    feed = []
    with llm.override_model(model), activity.report_to(lambda *e: feed.append(e)):
        out = run(dataprep.run(ws))
    assert ws.committed and [o.op for o in ws.ops] == ["clean_text"] and len(ws.current) == 132
    assert not ws.current["input"].str.contains("<p>|&nbsp;").any()
    assert "HTML" in out.rationale

    # лента: каждый вызов инструмента — строкой с аргументами, его итог — по ключу вызова, «думает» перед запросами
    steps = {key: text for kind, who, text, key in feed if kind == "step"}
    results = {key: text for kind, who, text, key in feed if kind == "result"}
    assert all(who == "A2 · подготовка данных" for _, who, _, _ in feed)
    assert sum(kind == "live" for kind, *_ in feed) == 8 and feed[-1][0] == "idle"
    by_text = [(steps[k], results.get(k)) for k in steps]
    assert ("проверяет цели качества", "не выполнено: мусор") in by_text
    assert ("применяет чистку текста", "строк 132 → 132") in by_text
    assert ("проверяет цели качества", "все цели выполнены") in by_text
    assert sum(1 for t, r in by_text if t == "фиксирует рецепт" and r and r.startswith("отклонено")) == 2
    assert ("фиксирует рецепт", "рецепт принят") in by_text


def test_dataprep_waiver_and_undo(cfg, tmp_path, monkeypatch):
    from factory.agents import dataprep
    from factory.prep import quality
    monkeypatch.setattr(quality.lang, "language_share", lambda texts: {"ru": 1.0})
    ws = _dirty_workspace(cfg, tmp_path)
    model = scripted(
        lambda m, i: tool("filter", field="label", operator="==", value="neg", apply=True),  # ошибочно выкинул класс
        lambda m, i: tool("undo"),
        lambda m, i: tool("commit_recipe", rationale="оставляю как есть",
                          waivers={"junk_text": "HTML в отзывах — часть реальных данных площадки, модель должна его видеть"}),
        lambda m, i: final(i, rationale="Данные оставлены без изменений."),
    )
    with llm.override_model(model):
        out = run(dataprep.run(ws))
    assert ws.ops == [] and len(ws.current) == 132 and "junk_text" in ws.waivers
    assert "Сознательно не выполнено" in out.rationale


def test_monitor_limit_exceeded_keeps_actions():
    from factory.agents import monitor
    live = FakeLive()
    deps = monitor.MonitorDeps(task_type="t", primary_metric="macro_f1", live=live, eval_steps=50)
    steps = [lambda m, i: tool("set_grad_clip", max_norm=0.5, reason="взрыв градиента")]
    steps += [lambda m, i: tool("get_recent_metrics", n=5)] * 30  # зацикливается на чтении
    with llm.override_model(scripted(*steps)):
        v = run(monitor.review(deps, "аномалии", []))
    assert live.sent[0][1] == {"kind": "grad_clip", "max_grad_norm": 0.5}
    assert v.verdict == "suspicious" and "взрыв градиента" in v.explanation  # итог собран из уже сделанных действий
    assert deps.observations == 1  # чтение — один раз за проверку: повторные вызовы скрыты, а не читаются до лимита


def test_dataprep_noop_operations_not_recorded(cfg, tmp_path):
    from factory.agents import dataprep
    ws = _dirty_workspace(cfg, tmp_path)
    with llm.override_model(scripted(lambda m, i: tool("dedupe", mode="exact", apply=True),
                                     lambda m, i: tool("dedupe", mode="exact", apply=True),
                                     lambda m, i: tool("commit_recipe", rationale="ок",
                                                       waivers={"junk_text": "проверяется в другом тесте, здесь не важно",
                                                                "language": "проверяется в другом тесте, здесь не важно",
                                                                "near_duplicates": "проверяется в другом тесте, здесь не важно"}),
                                     lambda m, i: final(i, rationale="ок"))):
        run(dataprep.run(ws))
    assert ws.ops == []  # в выборке нет точных дублей — оба вызова ничего не меняют


def test_monitor_stop_is_outside_the_adjustment_limit():
    """A4 решил, что остановить обучение «нельзя» (одна поправка за проверку, лимит вмешательств), хотя ни одной
    поправки не делал: остановка не поправка — доступна, пока разрешена настройками, независимо от остатка лимита."""
    import asyncio
    from types import SimpleNamespace

    from factory.agents.monitor import MonitorDeps, _stopping
    tool = object()

    def visible(**kw):
        deps = MonitorDeps(task_type="qa", primary_metric="token_f1", **kw)
        return asyncio.run(_stopping(SimpleNamespace(deps=deps), tool)) is tool
    assert visible()
    assert visible(interventions_left=0)  # лимит поправок исчерпан — остановка всё равно возможна
    assert not visible(allowed_actions=["continue", "patch"])  # запрещена настройками
    deps = MonitorDeps(task_type="qa", primary_metric="token_f1", interventions_left=0)
    deps.intervened = True
    assert asyncio.run(_stopping(SimpleNamespace(deps=deps), tool)) is tool  # и после поправки в этой проверке


def test_monitor_stop_with_exhausted_limit_is_applied_and_free():
    from factory.agents import monitor
    live = FakeLive()
    deps = monitor.MonitorDeps(task_type="qa", primary_metric="token_f1", interventions_left=0, live=live, eval_steps=50)
    model = scripted(
        lambda m, i: tool("get_val_curve"),
        lambda m, i: tool("stop_training", reason="переобучение: val хуже пика две оценки подряд"),
        lambda m, i: final(i, verdict="broken", explanation="Остановил: лучший адаптер сохранён."),
    )
    with llm.override_model(model):
        v = run(monitor.review(deps, "плановая проверка", []))
    assert live.sent == [("stop_and_save", None, "stop_early")] and v.verdict == "broken"
    assert deps.interventions_left == 0 and not deps.intervened


def test_monitor_reads_each_tool_once():
    """A4 вызывал кривую val по 4–5 раз подряд: повторное чтение возвращает то же самое и скрыто."""
    import asyncio
    from types import SimpleNamespace

    from factory.agents.monitor import MonitorDeps, _observing, get_val_curve
    tool = SimpleNamespace(name="get_val_curve")
    deps = MonitorDeps(task_type="qa", primary_metric="token_f1", live=FakeLive())
    ctx = SimpleNamespace(deps=deps)
    assert asyncio.run(_observing(ctx, tool)) is tool
    assert get_val_curve(ctx)
    assert asyncio.run(_observing(ctx, tool)) is None
    assert asyncio.run(_observing(ctx, SimpleNamespace(name="get_training_config"))).name == "get_training_config"


def test_monitor_one_intervention_per_review_and_none_after_limit():
    """A4 за одну проверку трижды снизил lr (×0.125), а потом раз за разом упирался в лимит: вмешательства
    видны только до первого в проверке и пока лимит не исчерпан."""
    import asyncio
    from types import SimpleNamespace

    from factory.agents.monitor import MonitorDeps, _intervening
    tool = object()
    deps = MonitorDeps(task_type="qa", primary_metric="token_f1")
    ctx = SimpleNamespace(deps=deps)
    assert asyncio.run(_intervening(ctx, tool)) is tool
    deps.intervened = True
    assert asyncio.run(_intervening(ctx, tool)) is None
    assert asyncio.run(_intervening(SimpleNamespace(deps=MonitorDeps(task_type="qa", primary_metric="f",
                                                                     interventions_left=0)), tool)) is None
