"""Бэкенд веб-чата: REST для сообщений и карточек, SSE — поток состояния сессии и событий обучения."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel

from factory.core.storage import atomic_write_text, read_json, read_jsonl
from factory.core.registry import Registry
from factory.orchestrator import Orchestrator
from factory.runs import RunPaths

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class NewSession(BaseModel):
    mode: str = "auto"


class Message(BaseModel):
    text: str


class Action(BaseModel):
    card_id: str | None = None
    action: str
    payload: dict[str, Any] | None = None


class LocalModelRequest(BaseModel):
    path: str
    languages: list[str] = ["multi"]
    license: str = ""
    commercial_ok: bool = False


class ModelSourceRequest(BaseModel):
    source: Literal["local", "hf"]


def create_app(orch: Orchestrator | None = None) -> FastAPI:
    orch = orch or Orchestrator()
    source_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await orch.recover()
        except Exception:
            logger.exception("Восстановление после старта не удалось")
        catalog = asyncio.create_task(orch.refresh_models())  # фоном: HF Hub и чтение карточек — минуты
        yield
        catalog.cancel()

    app = FastAPI(title="LoRA factory", lifespan=lifespan)
    app.state.orch = orch

    def state_json(sid: str) -> dict:
        try:
            return orch.load(sid).model_dump(mode="json")
        except KeyError as e:
            raise HTTPException(404, str(e))

    async def guarded(coro):
        try:
            return await coro
        except (ValueError, RuntimeError, KeyError) as e:
            raise HTTPException(400, str(e))

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/sessions")
    def sessions():
        return orch.list_sessions()

    @app.post("/api/sessions")
    def new_session(body: NewSession):
        return orch.create_session(body.mode).model_dump(mode="json")

    @app.get("/api/sessions/{sid}")
    def get_session(sid: str):
        return state_json(sid)

    @app.post("/api/sessions/{sid}/message")
    async def message(sid: str, body: Message):
        await guarded(orch.on_message(sid, body.text))
        return state_json(sid)

    @app.post("/api/sessions/{sid}/action")
    async def action(sid: str, body: Action):
        await guarded(orch.on_action(sid, body.card_id, body.action, body.payload))
        return state_json(sid)

    @app.post("/api/sessions/{sid}/settings")
    async def settings(sid: str, body: dict[str, Any]):
        await guarded(orch.update_settings(sid, body))
        return state_json(sid)

    @app.post("/api/sessions/{sid}/upload")
    async def upload(sid: str, request: Request, filename: str):
        data = await request.body()
        if not data:
            raise HTTPException(400, "Пустой файл")
        await guarded(orch.on_upload(sid, filename, data))
        return state_json(sid)

    @app.get("/results")
    def results_page():
        return FileResponse(STATIC / "results.html")

    @app.get("/api/hardware")
    def hardware():
        """Характеристики машины и пригодность моделей реестра (LoRA / QLoRA / не влезает, время)."""
        from factory.hparams.resources import rank_models
        hw = orch.hardware()
        models = [m for m in orch.registry.models.values()
                  if (m.source == "local") == orch.cfg.offline]
        return {"hardware": hw.model_dump(), "models": rank_models(models, hw)}

    @app.get("/api/model-source")
    def model_source():
        return {"source": "local" if orch.cfg.offline else "hf", "models_dir": str(orch.cfg.models_dir or "")}

    @app.put("/api/model-source")
    async def set_model_source(body: ModelSourceRequest):
        async with source_lock:
            if any(not task.done() for task in orch.tasks.values()):
                raise HTTPException(409, "Дождитесь завершения текущей операции перед переключением источника моделей")
            offline = body.source == "local"
            if orch.cfg.offline != offline:
                atomic_write_text(orch.cfg.data_dir / "model_source.json", body.model_dump_json())
                orch.cfg.offline = offline
                os.environ["FACTORY_OFFLINE"] = "1" if offline else "0"
                orch.registry = Registry.load(orch.cfg)
                if not offline:
                    asyncio.create_task(orch.refresh_models(force=True))
        return {"source": body.source, "models_dir": str(orch.cfg.models_dir or "")}

    @app.post("/api/models/refresh")
    async def refresh_models():
        """Заново подобрать модели с HF Hub под эту машину (не дожидаясь недельного срока кэша)."""
        return {"models": await orch.refresh_models(force=True)}

    @app.get("/api/models/local")
    def local_models():
        return [m.model_dump(mode="json") for m in orch.registry.models.values() if m.source == "local"]

    @app.post("/api/models/local")
    async def register_model(body: LocalModelRequest):
        from factory.core.local_models import register_local_model
        try:
            model = await asyncio.to_thread(register_local_model, orch.cfg, Path(body.path),
                                             languages=body.languages, license=body.license,
                                             commercial_ok=body.commercial_ok)
        except (OSError, ValueError) as e:
            raise HTTPException(400, str(e)) from e
        orch.registry.add_models([model])
        return model.model_dump(mode="json")

    @app.post("/api/datasets/merge", status_code=501)
    def merge_datasets(body: dict[str, Any]):
        from factory.prep.merge import MergeRequest, merge_datasets
        request = MergeRequest.model_validate(body)
        try:
            merge_datasets(request)
        except NotImplementedError as e:
            raise HTTPException(501, str(e)) from e

    @app.get("/api/runs")
    def runs():
        """Реестр запусков: постановка, датасет, конфигурация, метрики val/test, адаптер."""
        from factory.hparams.history import iter_manifests
        out = []
        for m in iter_manifests(orch.cfg.runs_dir):
            report = read_json(RunPaths(orch.cfg.runs_dir / m.run_id).eval_report) or {}
            out.append({"run_id": m.run_id, "status": m.status, "created_at": m.created_at, "task_type": m.spec.task_type,
                        "description": m.spec.description, "base_model": m.spec.base_model_id, "dataset": m.dataset.hf_id,
                        "backend": m.config.backend, "quantization": m.config.quantization, "rounds": len(m.rounds),
                        "primary_metric": m.spec.primary_metric, "val": m.final_val_metrics,
                        "test": m.test_metrics or report.get("metrics", {}),
                        "baselines_test": report.get("baselines", {}), "latency_ms": report.get("latency_ms"),
                        "adapter_path": m.adapter_path, "restart_of": m.restart_of_run_id})
        return sorted(out, key=lambda r: str(r["created_at"]), reverse=True)

    @app.get("/api/runs/{run_id}")
    def run(run_id: str):
        paths = RunPaths(orch.cfg.runs_dir / run_id)
        if not paths.manifest.exists():
            raise HTTPException(404, "Запуск не найден")
        return {"manifest": read_json(paths.manifest), "eval_report": read_json(paths.eval_report)}

    @app.get("/api/sessions/{sid}/events", response_class=EventSourceResponse)
    async def events(sid: str, request: Request):
        """state — при изменении session.json; metric/event — новые строки metrics.jsonl/events.jsonl запуска."""
        version = -1
        run_id = None
        m_off = e_off = 0
        while not await request.is_disconnected():
            try:
                st = orch.load(sid)
            except KeyError:
                yield ServerSentEvent(event="error", data={"detail": "session not found"})
                return
            if st.version != version:
                version = st.version
                yield ServerSentEvent(event="state", data=st.model_dump(mode="json"), id=str(version))
            if st.run_id != run_id:
                run_id, m_off, e_off = st.run_id, 0, 0
                yield ServerSentEvent(event="run", data={"run_id": run_id})
            if run_id:
                paths = RunPaths(orch.cfg.runs_dir / run_id)
                metrics, m_off = read_jsonl(paths.metrics, m_off)
                if metrics:
                    yield ServerSentEvent(event="metrics", data=metrics)
                evs, e_off = read_jsonl(paths.events, e_off)
                if evs:
                    yield ServerSentEvent(event="events", data=evs)
            await asyncio.sleep(1.0)

    return app
