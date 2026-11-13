"""Регрессии, найденные ревью."""
import asyncio

import pytest

from factory.core.contracts import (
    DatasetCandidate, LoraParams, OptimParams, Recipe, RunConfig, ScheduleParams, SessionSettings, Seeds,
)
from factory.core.storage import read_json, write_json
from factory.runs import create_run, reuse_key


def _config(seed=1):
    return RunConfig(backend="peft_lora", base_model_id="Qwen/Qwen3-0.6B", max_seq_len=128, lora=LoraParams(r=8, alpha=16),
                     optim=OptimParams(lr=1e-4), schedule=ScheduleParams(epochs=1), seeds=Seeds(split_seed=1, train_seed=seed))


def test_reuse_key_ignores_random_train_seed(cls_spec):
    ds, recipe = DatasetCandidate(hf_id="x"), Recipe(dataset_id="x", split_seed=1)
    assert reuse_key(cls_spec, ds, recipe, _config(1)) == reuse_key(cls_spec, ds, recipe, _config(999))
    other = _config(1).model_copy(update={"optim": OptimParams(lr=2e-4)})
    assert reuse_key(cls_spec, ds, recipe, _config(1)) != reuse_key(cls_spec, ds, recipe, other)


def test_restart_passes_failed_training_to_new_hparam_selection(cfg, registry, cls_spec, monkeypatch, data_dir):
    from types import SimpleNamespace
    from factory.core.contracts import DataProfile, LengthStats, RoundReport
    from factory.core.state import Step
    from factory.hparams import guard, heuristic, strategy
    from factory.hparams.pilot import PilotReport
    from factory.hparams.retuning import load_previous_attempts, training_signature
    from factory.orchestrator import Orchestrator
    from factory.runs import load_manifest, save_manifest
    from factory.training import launcher

    orch = Orchestrator(cfg, registry)
    monkeypatch.setattr(orch, "hardware", lambda: SimpleNamespace(usable_vram_gb=12.0))
    st = orch.create_session("expert")
    st.spec, st.dataset = cls_spec, DatasetCandidate(hf_id="x")
    recipe = Recipe(dataset_id="x", split_seed=1, profile_after=DataProfile(
        n_rows=2000, input_tokens_est=LengthStats(p95=80, p99=120), target_tokens_est=LengthStats(p95=3, p99=4)))
    session = orch.session_dir(st.session_id)
    (session / "data").mkdir(exist_ok=True)
    write_json(session / "recipe.json", recipe)
    original_ctx = orch._hparam_context(st, recipe, 42)
    config = guard.assemble(heuristic.suggest(original_ctx, None), original_ctx, "heuristic")
    paths, manifest = create_run(cfg.runs_dir, st.spec, st.dataset, recipe, config, session, st.session_id)
    manifest.status = "round_review"
    manifest.rounds = [RoundReport(round=1, epochs_done=3, val_metrics={"macro_f1": 0.2},
                                   diagnosis="Недостаточное качество", recommendation="restart")]
    manifest.test_metrics = {"macro_f1": 0.99}  # test запрещён как вход повторного подбора
    save_manifest(paths, manifest)
    st.run_id, st.run_ids, st.step = paths.run_id, [paths.run_id], Step.ROUND_REVIEW
    st.recipe_path = str(session / "recipe.json")
    orch.save(st)

    async def baseline(run_paths, *args):
        write_json(run_paths.baselines_val, {})
        return {"status": "ok"}

    async def fake_pilot(run_paths, config, *args):
        return PilotReport(status="ok", config=config)

    async def start(sid):
        await orch._run_hparams(sid)

    monkeypatch.setattr(launcher, "run", baseline)
    monkeypatch.setattr(strategy, "run_pilot", fake_pilot)
    monkeypatch.setattr(orch, "_start_hparams", start)
    asyncio.run(orch._round_decision(st.session_id, "restart", None, "user"))
    after = orch.load(st.session_id)
    assert after.step == Step.CONFIG_CONFIRM and after.run_id != paths.run_id
    assert after.restart_of_run_id == paths.run_id
    assert training_signature(after.pending_config) != training_signature(config)
    attempts = load_previous_attempts(cfg.runs_dir, after.restart_of_run_id)
    assert attempts[0].rounds[0].val_metrics == {"macro_f1": 0.2}
    assert "test_metrics" not in str(attempts[0].summary()) and "0.99" not in str(attempts[0].summary())
    assert load_manifest(paths).status == "interrupted"

    # Ещё один неудачный запуск: не возвращаемся ни к первой, ни ко второй конфигурации.
    from factory.runs import RunPaths
    second_paths = RunPaths(cfg.runs_dir / after.run_id)
    second = load_manifest(second_paths)
    second.rounds = [RoundReport(round=1, epochs_done=2, val_metrics={"macro_f1": 0.25}, recommendation="restart")]
    second.status = "round_review"
    save_manifest(second_paths, second)
    after.step = Step.ROUND_REVIEW
    orch.save(after)
    asyncio.run(orch._round_decision(st.session_id, "restart", None, "user"))
    third = orch.load(st.session_id)
    assert training_signature(third.pending_config) not in {training_signature(config), training_signature(second.config)}
    assert {a.run_id for a in load_previous_attempts(cfg.runs_dir, third.restart_of_run_id)} == {paths.run_id, second_paths.run_id}


@pytest.fixture
def run_dir(data_dir, cls_spec, cfg):
    session = data_dir / "sessions" / "s"
    (session / "data").mkdir(parents=True)
    paths, _ = create_run(cfg.runs_dir, cls_spec, DatasetCandidate(hf_id="x"), Recipe(dataset_id="x", split_seed=1),
                          _config(), session, "s")
    return paths


class _Proc:
    pid = 4242

    def poll(self):
        return 0


def test_new_round_clears_old_commands_and_heartbeat(run_dir, cfg, registry, monkeypatch):
    from factory.training import supervisor
    write_json(run_dir.control, {"commands": [{"id": "old", "cmd": "stop_and_save"}]})
    run_dir.heartbeat.write_text("0")
    seen = {}

    def fake_spawn(paths, mode, status_file, *extra):
        seen["control_exists"] = paths.control.exists()
        seen["heartbeat_exists"] = paths.heartbeat.exists()
        write_json(status_file, {"status": "ok", "global_step": 1})
        return _Proc()
    monkeypatch.setattr(supervisor.launcher, "spawn", fake_spawn)
    sup = supervisor.Supervisor(run_dir, cfg, registry, SessionSettings(), notify=lambda t: None, pilot_tps=None, vram_gb=None)
    out = asyncio.run(sup.run_round(supervisor.RoundPlan(round_k=2, epochs=1), _config()))
    assert out.status == "ok"
    assert seen == {"control_exists": False, "heartbeat_exists": False}


def test_reattach_watches_running_process(run_dir, cfg, registry, monkeypatch):
    from factory.training import supervisor
    alive = iter([True, True, False])
    monkeypatch.setattr(supervisor.launcher, "pid_alive", lambda pid: next(alive))
    monkeypatch.setattr(supervisor.launcher, "spawn", lambda *a: pytest.fail("при подключении нельзя запускать новый процесс"))
    rdir = run_dir.round_dir(1)
    rdir.mkdir(parents=True)
    write_json(rdir / "status.json", {"status": "ok", "global_step": 7})
    cfg.training.poll_interval_s = 0.01
    sup = supervisor.Supervisor(run_dir, cfg, registry, SessionSettings(), notify=lambda t: None, pilot_tps=None, vram_gb=None)
    out = asyncio.run(sup.run_round(supervisor.RoundPlan(round_k=1, epochs=1), _config(), attach_pid=4242))
    assert out.status == "ok" and out.result["global_step"] == 7


def test_dead_process_without_final_status_is_an_error(run_dir, cfg, registry, monkeypatch):
    from factory.training import supervisor

    def fake_spawn(paths, mode, status_file, *extra):
        write_json(status_file, {"status": "running", "pid": 1})  # упал, не дописав итог
        return _Proc()
    monkeypatch.setattr(supervisor.launcher, "spawn", fake_spawn)
    sup = supervisor.Supervisor(run_dir, cfg, registry, SessionSettings(), notify=lambda t: None, pilot_tps=None, vram_gb=None)
    out = asyncio.run(sup.run_round(supervisor.RoundPlan(round_k=1, epochs=1), _config()))
    assert out.status == "failed" and "без итогового статуса" in out.result["message"]


def test_auto_confirmation_closes_the_card(cfg, registry):
    from factory.core.state import Card
    from factory.orchestrator import Orchestrator
    orch = Orchestrator(cfg, registry)
    st = orch.create_session("auto")
    st.say("Постановка", card=Card(kind="spec", blocking=False))
    orch._auto(st, "spec", "режим Авто: постановка однозначна")
    card = next(i.card for i in st.items if i.card)
    assert card.resolved and card.resolution == {"action": "auto"}  # кнопки «Подтвердить» на карточке больше нет


def test_chat_reply_closes_cards_waiting_for_decision(cfg, registry, cls_spec, monkeypatch):
    from factory.core.state import Card, Step
    from factory.orchestrator import Orchestrator
    started = []

    async def no_background(self, sid, label, fn):
        started.append(label)
    monkeypatch.setattr(Orchestrator, "_background", no_background)
    orch = Orchestrator(cfg, registry)
    st = orch.create_session("auto")
    st.spec, st.step = cls_spec, Step.DATASET_CHOICE
    st.say("Подходящих датасетов не нашлось.", card=Card(kind="dataset_choice", payload={"candidates": []}))
    orch.save(st)
    st = asyncio.run(orch.on_message(st.session_id, "Поищи отзывы на товары"))
    # старая карточка не должна блокировать автоподтверждения следующего круга
    assert st.step == Step.INTAKE and not st.open_cards() and started == ["диалог"]
    assert next(i.card for i in st.items if i.card).resolution == {"action": "none_fit"}


def test_label_renames_and_label_filters_reach_val_and_test():
    import pandas as pd
    from factory.core.contracts import DedupeOp, FilterOp, MapLabelsOp
    from factory.prep.ops import apply_label_schema
    recipe = Recipe(dataset_id="x", split_seed=1, ops=[DedupeOp(), MapLabelsOp(mapping={"pos": "позитив"}),
                                                        FilterOp(field="label", operator="!=", value="мусор")])
    val = pd.DataFrame({"input": ["a", "a", "b", "c"], "target": ["pos", "pos", "neg", "мусор"], "synthetic": False})
    # метки — как в train; чистка (здесь дедупликация) val не касается
    assert apply_label_schema(val, recipe)["target"].tolist() == ["позитив", "позитив", "neg"]


def test_pilot_ignores_status_of_previous_pilot(run_dir, cfg, monkeypatch):
    from factory.hparams import pilot
    run_dir.pilot.mkdir(parents=True, exist_ok=True)
    write_json(run_dir.pilot / "status_1.json", {"status": "ok", "tokens_per_sec": 1.0})  # другая конфигурация

    async def fake_run(paths, mode, status_file, *extra):
        write_json(status_file, {"status": "ok", "tokens_per_sec": 500.0})
        return read_json(status_file)
    monkeypatch.setattr(pilot.launcher, "run", fake_run)
    assert asyncio.run(pilot.run_pilot(run_dir, _config(), 100, cfg.hparams.pilot)).ok
    assert [p.name for p in run_dir.pilot.glob("status_*.json")] == ["status_0.json"]


def test_planned_monitor_checks_continue_after_window_fills(run_dir, cfg, registry, monkeypatch):
    from factory.core.contracts import MetricPoint
    from factory.core.storage import append_jsonl
    from factory.training import supervisor
    cfg.training.poll_interval_s, cfg.training.monitor_cooldown_steps, cfg.training.monitor_check_every_evals = 0.0, 0, 3
    status_file = run_dir.round_dir(1) / "status.json"
    status_file.parent.mkdir(parents=True)
    step, polls = 0, iter(range(12))

    def alive():  # «процесс»: между опросами — 60 точек train loss и одна оценка на val
        nonlocal step
        if next(polls, None) is None:
            write_json(status_file, {"status": "ok"})
            return False
        for _ in range(60):
            step += 1
            append_jsonl(run_dir.metrics, MetricPoint(step=step, train_loss=1.0))
        append_jsonl(run_dir.metrics, MetricPoint(step=step, val_loss=1.0, val_metrics={"macro_f1": 0.5}))
        return True
    checks = []

    async def fake_consult(self, soft, window, config, trigger):
        checks.append(trigger)
    monkeypatch.setattr(supervisor.Supervisor, "_consult_monitor", fake_consult)
    sup = supervisor.Supervisor(run_dir, cfg, registry, SessionSettings(), notify=lambda t: None, pilot_tps=None, vram_gb=None)
    asyncio.run(sup._watch(alive, status_file, supervisor.RoundPlan(1, 1), _config()))
    assert len(checks) == 4  # 12 оценок, проверка каждые 3; окно в 200 точек вмещает только ~3 оценки


def test_unconfirmed_command_is_not_repeated(run_dir, cfg, registry):
    """В полном прогоне A4 трижды послал «lr × 0.5», пока процесс не подтверждал (цикл обучения уже закончился):
    повтор неподтверждённой команды не отправляется и не тратит вмешательство; после конца цикла — не отправляется."""
    from factory.core.contracts import MonitorDecision
    from factory.training import supervisor
    cfg.training.monitor_ack_timeout_s = 0.1
    sup = supervisor.Supervisor(run_dir, cfg, registry, SessionSettings(), notify=lambda t: None, pilot_tps=None, vram_gb=None)
    decision = MonitorDecision(verdict="suspicious", action="patch", patch={"kind": "lr_scale", "factor": 0.5}, explanation="плато")
    first = asyncio.run(sup.send_and_confirm("patch", {"kind": "lr_scale", "factor": 0.5}, decision))
    second = asyncio.run(sup.send_and_confirm("patch", {"kind": "lr_scale", "factor": 0.5}, decision))
    assert first["sent"] and not first["applied"]
    assert not second["sent"] and len(read_json(run_dir.control)["commands"]) == 1
    sup.pending.clear()
    sup.loop_ended = True
    assert not asyncio.run(sup.send_and_confirm("eval_now", None, decision))["sent"]


def test_drop_answers_stored_as_strings_are_parsed():
    """AvitoTech/ru_drop: ответы — строки с repr словаря DROP; целью была запись всего словаря."""
    import pandas as pd
    from factory.core.contracts import MapSchemaOp
    from factory.prep.schema import to_task_frame
    df = pd.DataFrame({
        "passage": ["Его застрелили в спину.", "Собрали 8800 подписей."], "question": ["Как погиб?", "Сколько подписей?"],
        "validated_answers": [
            "{'number': ['', ''], 'date': [{'day': '', 'month': '', 'year': ''}], 'spans': [['застрелен в спину'], ['застрелен в спину']]}",
            "{'number': ['8800'], 'date': [{'day': '', 'month': '', 'year': ''}], 'spans': [[]]}"]})
    out = to_task_frame(df, MapSchemaOp(kind="qa", input_column="passage", target_column="validated_answers",
                                        question_column="question"))
    assert out["target"].tolist() == ["застрелен в спину", "8800"]


def test_short_numeric_answers_are_not_junk():
    import pandas as pd
    from factory.prep.quality import junk_mask
    answers = pd.Series(["8800", "1995 г.", "5", "", "<b>спам</b>", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11"])
    assert junk_mask(answers, answers=True).tolist() == [False, False, False, True, True, True]
    assert junk_mask(pd.Series(["5", "!!!"])).tolist() == [True, True]  # вход из одних цифр/знаков — по-прежнему мусор


def test_auto_mode_does_not_pick_weak_dataset(cfg, registry, cls_spec):
    """Все кандидаты ~40 из 100 и «не подходит» — раньше Авто всё равно брал первый."""
    from factory.core.state import Step
    from factory.orchestrator import Orchestrator
    orch = Orchestrator(cfg, registry)
    st = orch.create_session("auto")
    st.spec, st.step = cls_spec, Step.SEARCH
    orch._present(st, [DatasetCandidate(hf_id="weak", score=40.0)], None, "Найденные датасеты не подходят")
    assert [c.kind for c in st.open_cards()] == ["dataset_choice"]
    st2 = orch.create_session("auto")
    st2.spec, st2.step = cls_spec, Step.SEARCH
    orch._present(st2, [DatasetCandidate(hf_id="user:mine.csv", source="user", score=0.0)], None, "")
    assert not st2.open_cards()  # свой файл пользователя Авто берёт без порога


def test_dataprep_notes_repeated_observation(cfg, tmp_path):
    from factory.agents.dataprep import PrepWorkspace
    from factory.core.contracts import DedupeOp
    import pandas as pd
    df = pd.DataFrame({"_id": ["1", "2"], "input": ["а", "б"], "target": ["x", "y"], "synthetic": False})
    ws = PrepWorkspace(sample=df, family="generation", language="ru", labels=None, task_description="t", cfg=cfg.dataprep,
                       mode="auto", synth_dir=tmp_path, seed=0, findings=[], full_train_rows=2)
    assert ws.repeat_note("check_quality") == {}
    assert "не менялись" in ws.repeat_note("check_quality")["note"]
    ws.record(DedupeOp(), df)
    assert ws.repeat_note("check_quality") == {}


def test_checkpoint_offsets_follow_the_process_that_saved_it(tmp_path):
    """Номер чекпоинта — шаг внутри процесса. Раунд продолжался после OOM с шага 238: его чекпоинт-238 — это
    глобальный шаг 476, а не 238 (иначе восстановление пересчитает полэпохи заново)."""
    import json
    import os
    import time

    from factory.training.supervisor import checkpoint_offsets

    ck = tmp_path / "checkpoint-238"
    ck.mkdir()
    (ck / "trainer_state.json").write_text(json.dumps({"global_step": 238, "epoch": 0.5, "log_history": []}))
    now = time.time()
    os.utime(ck, (now, now))
    first = {"type": "round_started", "time": now - 7200, "epoch_offset": 0.0}
    second = {"type": "round_started", "time": now - 3600, "step_offset": 238, "epoch_offset": 0.5}
    later = {"type": "round_started", "time": now + 10, "step_offset": 476, "epoch_offset": 1.0}  # после сохранения
    assert checkpoint_offsets(ck, [first, second, later], 0, 0.0) == (238, 0.5)
    old = {"type": "round_started", "time": now - 3600, "epoch_offset": 0.5}  # события до step_offset
    assert checkpoint_offsets(ck, [first, old], 0, 0.0) == (238, 0.5)
    assert checkpoint_offsets(ck, [], 0, 0.0) == (0, 0.0)


def test_all_modules_compile():
    """Рабочий процесс обучения тесты не импортируют: синтаксическая ошибка в нём ломала baseline/обучение
    (процесс падал при запуске, «завершился с кодом 1 без статуса»), а набор тестов оставался зелёным."""
    import py_compile
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "factory"
    for path in src.rglob("*.py"):
        py_compile.compile(str(path), doraise=True)


def test_recover_uses_latest_attempt_checkpoint(run_dir, cfg, registry, monkeypatch):
    from factory.core.state import SessionState, Step
    from factory.core.storage import append_jsonl
    from factory.orchestrator import Orchestrator
    from factory.runs import load_manifest, save_manifest
    from factory.training.checkpoints import new_attempt

    orch = Orchestrator(cfg, registry)
    manifest = load_manifest(run_dir)
    manifest.status, manifest.pid = "running", None
    save_manifest(run_dir, manifest)
    orch.save(SessionState(session_id="s", step=Step.TRAINING, busy="training", run_id=run_dir.run_id))
    rdir = run_dir.round_dir(1)
    old = new_attempt(rdir) / "checkpoint-100"
    write_json(old / "trainer_state.json", {"global_step": 100, "epoch": 0.5})
    new = new_attempt(rdir) / "checkpoint-20"
    write_json(new / "trainer_state.json", {"global_step": 20, "epoch": 0.1})
    append_jsonl(run_dir.events, {"type": "round_started", "time": new.stat().st_mtime - 1,
                                 "step_offset": 100, "epoch_offset": 0.5})
    captured = []

    async def training(sid, plan, config):
        captured.append(plan)

    async def background(sid, label, fn):
        await fn()

    monkeypatch.setattr(orch, "_run_training", training)
    monkeypatch.setattr(orch, "_background", background)
    asyncio.run(orch.recover())
    assert len(captured) == 1
    assert captured[0].resume_from == new
    assert captured[0].step_offset == 120
    assert captured[0].epoch_offset == pytest.approx(0.6)
