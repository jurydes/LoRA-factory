"""Сквозной сценарий через оркестратор: фейковая LLM (FunctionModel) за всех агентов, обучение — настоящими
подпроцессами на крошечной модели. INTAKE → … → DONE в режиме Авто."""
import asyncio
import random

import pandas as pd
import pytest
import yaml
from pydantic_ai import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from factory.core import llm
from factory.core.state import Step

pytestmark = pytest.mark.integration


def fake_llm(model_id: str):
    def fn(messages, info: AgentInfo):
        tools = {t.name for t in info.function_tools}
        returned = {p.tool_name: p.content for m in messages for p in getattr(m, "parts", []) if isinstance(p, ToolReturnPart)}
        last_prompt = next((p.content for m in reversed(messages) for p in getattr(m, "parts", [])
                            if getattr(p, "part_kind", "") == "user-prompt"), "")
        if not info.output_tools:  # текстовый выход: расширение запроса
            return ModelResponse(parts=[TextPart("Я решаю задачу классификации тональности отзывов на русском.")])
        names = {t.name: t for t in info.output_tools}

        def out(name_part, **args):
            name = next(n for n in names if name_part.lower() in n.lower()) if name_part else next(iter(names))
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])

        props = set(next(iter(names.values())).parameters_json_schema.get("properties", {}))
        if "search_datasets" in tools:  # A1
            if "search_datasets" in returned:
                return out("PresentDatasets", hf_ids=[returned["search_datasets"][0]["hf_id"]], message="Нашёл датасет.")
            if "Найди" in last_prompt:
                return ModelResponse(parts=[ToolCallPart(tool_name="search_datasets", args={})])
            return out("ProposeSpec", spec={"description": "Определить тональность отзыва", "task_type": "sentiment_analysis",
                                            "language": "ru", "labels": ["позитив", "негатив"], "base_model_id": model_id})
        if "relevance" in props:
            return out(None, **{k: {"score": 4, "reason": "ок"} for k in
                                ["relevance", "label_fit", "size", "text_quality", "domain_fit", "label_reliability"]},
                       why="Отзывы с разметкой тональности")
        if "verdict" in props:
            return out(None, verdict="ok", explanation="всё нормально")
        if "recommendation" in props:
            return out(None, diagnosis="Метрика {val.macro_f1} против baseline {baseline.majority.macro_f1}.",
                       recommendation="finish" if "ДОСТАТОЧНО" in last_prompt and "НЕДОСТАТОЧНО" not in last_prompt else "continue",
                       extra_epochs=1)
        if "diagnosis" in props:
            return out(None, diagnosis="На test {test.macro_f1}; лучший baseline {baseline.majority.macro_f1}.")
        raise AssertionError(f"неожиданный вызов: {props} {tools}")
    return FunctionModel(fn)


async def wait_for(orch, sid, steps, timeout=600):
    for _ in range(int(timeout / 0.5)):
        st = orch.load(sid)
        if st.step in steps and not st.busy:
            return st
        if st.step == Step.FAILED:
            raise AssertionError(f"FAILED: {st.error}\n" + "\n".join(i.text for i in st.items[-8:]))
        await asyncio.sleep(0.5)
    raise AssertionError(f"не дождались {steps}: {orch.load(sid).step}, busy={orch.load(sid).busy}")


def test_full_scenario_auto_mode(data_dir, tiny_registered, monkeypatch, tmp_path):
    (data_dir / "config.yaml").write_text(yaml.safe_dump({"training": {"poll_interval_s": 0.5, "num_eval_samples": 40},
                                                          "hparams": {"pilot": {"min_steps": 4, "max_steps": 6}}}))
    from factory.core import config
    config.get_config.cache_clear()

    rng = random.Random(0)
    rows = [(" ".join(rng.choices(["хороший", "отличный", "прекрасный", "товар"], k=6)) + f" {i}", "позитив") for i in range(250)]
    rows += [(" ".join(rng.choices(["плохой", "ужасный", "отвратительный", "товар"], k=6)) + f" {i}", "негатив") for i in range(250)]
    csv = tmp_path / "reviews.csv"
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(csv, index=False)

    from factory.orchestrator import Orchestrator
    from factory.search import funnel

    async def search_stub(self, sid):
        async def search(spec, extra):
            return [await funnel.evaluate_user_dataset(csv, spec, self.registry, self.cfg)]
        return search
    monkeypatch.setattr(Orchestrator, "_search_fn", search_stub)

    async def scenario():
        orch = Orchestrator()
        st = orch.create_session("auto")
        sid = st.session_id
        with llm.override_model(fake_llm(str(tiny_registered))):
            await orch.on_message(sid, "Хочу классифицировать отзывы на позитив и негатив")
            st = await wait_for(orch, sid, {Step.CONFIG_CONFIRM})
            kinds = [i.card.kind for i in st.items if i.card]
            assert {"spec", "dataset_choice", "prep", "config"} <= set(kinds)
            assert {a.card for a in st.auto_confirmations} >= {"spec", "dataset_choice", "prep"}
            # автоподтверждённые карточки закрыты — кнопок «Подтвердить»/«Выбрать» на них нет; ждёт только config
            waiting = [i.card.kind for i in st.items if i.card and not i.card.resolved and i.card.kind != "info"]
            assert waiting == ["config"], waiting
            assert any("пробный прогон" in a.text for a in st.activity)  # лента: рабочие процессы фазы
            config_card = next(i.card for i in reversed(st.items) if i.card and i.card.kind == "config")
            assert config_card.blocking and config_card.payload["pilot"]["status"] == "ok"
            await orch.on_action(sid, config_card.id, "confirm_config", {})
            st = await wait_for(orch, sid, {Step.DONE, Step.ROUND_REVIEW}, timeout=900)
            if st.step == Step.ROUND_REVIEW:  # §3.4: качество недостаточно — даже в Авто решает пользователь
                rc = next(i.card for i in reversed(st.items) if i.card and i.card.kind == "round")
                assert rc.blocking and not rc.payload["report"]["sufficient"]
                await orch.on_action(sid, rc.id, "round_finish")
                st = await wait_for(orch, sid, {Step.DONE}, timeout=900)
        result = next(i.card for i in reversed(st.items) if i.card and i.card.kind == "result")
        report = result.payload["report"]
        assert set(report["metrics"]) == {"macro_f1", "accuracy", "weighted_f1"} and "majority" in report["baselines"]
        assert "{" not in report["diagnosis"]  # плейсхолдеры подставлены кодом
        from factory.runs import RunPaths, load_manifest
        m = load_manifest(RunPaths(orch.cfg.runs_dir / st.run_id))
        assert m.status == "finished" and m.adapter_path and m.rounds
        assert m.rounds[-1].chosen_by == ("system" if m.rounds[-1].sufficient else "user")
        assert report.get("latency_ms", 0) > 0
        from factory.core.storage import read_jsonl
        pts = [p for p in read_jsonl(RunPaths(orch.cfg.runs_dir / st.run_id).metrics)[0] if p.get("train_loss") is not None]
        assert pts and all(p["max_step"] >= p["step"] and p["eta_s"] >= 0 for p in pts) and pts[-1]["eta_s"] < pts[0]["eta_s"] + 1
        assert m.test_metrics and m.final_val_metrics
        assert (orch.session_dir(sid) / "llm_calls.jsonl").exists()
        assert (orch.session_dir(sid) / "conversation.json").exists()
    asyncio.run(scenario())


def test_activity_feed_merge(data_dir):
    """«Думает» — временная строка; итог инструмента дописывается к его строке по ключу; лента ограничена."""
    from factory.orchestrator import ACTIVITY_KEEP, Orchestrator

    async def scenario():
        orch = Orchestrator()
        sid = orch.create_session("auto").session_id
        await orch._update(sid, lambda st: orch._set_busy(st, "готовлю данные"))
        for e in [("live", "A2", "думает", None), ("step", "A2", "проверяет цели качества", "k1"),
                  ("live", "A2", "думает (запрос 2)", None), ("result", "A2", "не выполнено: мусор", "k1")]:
            await orch._activity(sid, *e)
        st = orch.load(sid)
        assert [(a.text, a.result, a.live) for a in st.activity] == [("проверяет цели качества", "не выполнено: мусор", False)]
        await orch._activity(sid, "live", "A2", "думает (запрос 3)", None)
        await orch._activity(sid, "idle", "A2", "", None)
        assert len(orch.load(sid).activity) == 1
        for i in range(ACTIVITY_KEEP + 3):
            await orch._activity(sid, "step", "A2", f"шаг {i}", None)
        st = orch.load(sid)
        assert len(st.activity) == ACTIVITY_KEEP and st.activity[-1].text == f"шаг {ACTIVITY_KEEP + 2}" and st.busy_since
        await orch._update(sid, lambda st: (setattr(st, "busy", None), orch._set_busy(st, "подбираю гиперпараметры")))
        assert orch.load(sid).activity == []
    asyncio.run(scenario())


def test_api_smoke(data_dir):
    from fastapi.testclient import TestClient

    from factory.api.app import create_app
    with TestClient(create_app()) as client:
        st = client.post("/api/sessions", json={"mode": "expert"}).json()
        assert st["settings"]["mode"] == "expert" and st["step"] == "INTAKE"
        assert client.get(f"/api/sessions/{st['session_id']}").status_code == 200
        assert client.post(f"/api/sessions/{st['session_id']}/action", json={"action": "confirm_prep"}).status_code == 400
        assert client.get("/").status_code == 200 and client.get("/results").status_code == 200
        hwr = client.get("/api/hardware").json()
        assert "hardware" in hwr and any(m["id"] == "Qwen/Qwen3-0.6B" for m in hwr["models"])
        assert client.get("/api/runs").json() == []
        st2 = client.post(f"/api/sessions/{st['session_id']}/settings",
                          json={"limits": {"max_monitor_interventions": 1, "max_rounds": 2,
                                           "allowed_monitor_actions": ["continue", "patch"]}}).json()
        assert st2["settings"]["limits"]["max_rounds"] == 2 and "Лимиты мониторинга" in st2["items"][-1]["text"]
        assert any(s["session_id"] == st["session_id"] for s in client.get("/api/sessions").json())


def test_expert_mode_upload_edit_continue(data_dir, tiny_registered, monkeypatch, tmp_path):
    (data_dir / "config.yaml").write_text(yaml.safe_dump({"training": {"poll_interval_s": 0.5, "num_eval_samples": 40},
                                                          "hparams": {"pilot": {"min_steps": 4, "max_steps": 6}}}))
    from factory.core import config
    config.get_config.cache_clear()
    rng = random.Random(1)
    rows = [(" ".join(rng.choices(["хороший", "отличный", "прекрасный", "товар"], k=6)) + f" {i}", "Позитив") for i in range(250)]
    rows += [(" ".join(rng.choices(["плохой", "ужасный", "отвратительный", "товар"], k=6)) + f" {i}", "Негатив") for i in range(250)]
    data = pd.DataFrame(rows, columns=["review", "sentiment"]).to_csv(index=False).encode()

    from factory.orchestrator import Orchestrator

    def open_card(st, kind):
        return next(i.card for i in reversed(st.items) if i.card and i.card.kind == kind and not i.card.resolved)

    async def scenario():
        orch = Orchestrator()
        sid = orch.create_session("expert").session_id
        with llm.override_model(fake_llm(str(tiny_registered))):
            await orch.on_message(sid, "Классификация отзывов")
            st = await wait_for(orch, sid, {Step.SPEC_CONFIRM})
            assert open_card(st, "spec").blocking  # в режиме эксперта подтверждение обязательно
            await orch.on_upload(sid, "my_reviews.csv", data)  # подтверждение постановки со своим датасетом
            st = await wait_for(orch, sid, {Step.DATASET_CHOICE})
            cand = open_card(st, "dataset_choice").payload["candidates"][0]
            assert cand["source"] == "user" and cand["compat"]["label_map"] == {"Позитив": "позитив", "Негатив": "негатив"}
            await orch.on_action(sid, open_card(st, "dataset_choice").id, "choose_dataset", {"hf_id": cand["hf_id"]})
            st = await wait_for(orch, sid, {Step.PREP_CONFIRM})
            # эксперт правит рецепт: чистит текст и отсекает слишком короткие входы
            await orch.on_action(sid, open_card(st, "prep").id, "edit_recipe",
                                 {"ops": [{"op": "clean_text"}, {"op": "filter", "field": "input_len", "operator": ">", "value": 3}]})
            st = await wait_for(orch, sid, {Step.PREP_CONFIRM})
            ops = [o["op"] for o in open_card(st, "prep").payload["recipe"]["ops"]]
            assert ops == ["map_schema", "clean_text", "filter"]
            await orch.on_action(sid, open_card(st, "prep").id, "confirm_prep")
            st = await wait_for(orch, sid, {Step.CONFIG_CONFIRM})
            card = open_card(st, "config")
            assert card.payload["editable"]
            with pytest.raises(ValueError):  # guard: чужие модули отклоняются, карточка снова ждёт ответа
                bad = {"lora": dict(card.payload["config"]["lora"], target_modules=["evil"])}
                await orch.on_action(sid, card.id, "confirm_config", {"config": bad})
            st = orch.load(sid)
            card = open_card(st, "config")
            await orch.on_action(sid, card.id, "confirm_config",
                                 {"config": {"schedule": dict(card.payload["config"]["schedule"], epochs=1)}})
            st = await wait_for(orch, sid, {Step.ROUND_REVIEW}, timeout=600)
            await orch.on_action(sid, open_card(st, "round").id, "round_continue", {"epochs": 1})
            st = await wait_for(orch, sid, {Step.ROUND_REVIEW}, timeout=600)
            rounds = [i.card for i in st.items if i.card and i.card.kind == "round"]
            assert len(rounds) == 2 and rounds[-1].payload["report"]["round"] == 2
            await orch.on_action(sid, rounds[-1].id, "round_finish")
            st = await wait_for(orch, sid, {Step.DONE}, timeout=600)
        from factory.runs import RunPaths, load_manifest
        m = load_manifest(RunPaths(orch.cfg.runs_dir / st.run_id))
        assert [r.chosen_by for r in m.rounds] == ["user", "user"] and m.config.schedule.epochs == 1
        assert m.rounds[1].epochs_done > m.rounds[0].epochs_done
        assert not st.auto_confirmations
    asyncio.run(scenario())


def test_auto_mode_early_upload(data_dir, tiny_registered, monkeypatch):
    (data_dir / "config.yaml").write_text(yaml.safe_dump({"hparams": {"pilot": {"min_steps": 4, "max_steps": 6}},
                                                          "training": {"num_eval_samples": 40}}))
    from factory.core import config
    config.get_config.cache_clear()
    rows = [(f"хороший отличный товар номер {i}", "позитив") for i in range(200)]
    rows += [(f"плохой ужасный товар номер {i}", "негатив") for i in range(200)]
    data = pd.DataFrame(rows, columns=["text", "label"]).to_json(orient="records", lines=True, force_ascii=False).encode()

    from factory.orchestrator import Orchestrator

    async def no_search(self, sid):
        async def search(spec, extra):
            raise AssertionError("при загруженном датасете поиск по каталогу не нужен")
        return search
    monkeypatch.setattr(Orchestrator, "_search_fn", no_search)

    async def scenario():
        orch = Orchestrator()
        sid = orch.create_session("auto").session_id
        await orch.on_upload(sid, "reviews.jsonl", data)
        assert orch.load(sid).user_dataset_path
        with llm.override_model(fake_llm(str(tiny_registered))):
            await orch.on_message(sid, "Классификация отзывов, датасет приложил")
            st = await wait_for(orch, sid, {Step.CONFIG_CONFIRM})
        assert st.dataset.source == "user" and st.dataset.hf_id == "user:reviews.jsonl"
        assert {a.card for a in st.auto_confirmations} >= {"spec", "dataset_choice", "prep"}
    asyncio.run(scenario())
