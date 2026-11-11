"""Локальная модель и пользовательский датасет без доступа к Hugging Face."""
import asyncio
from pathlib import Path

import pytest

from factory.core.config import AppConfig
from factory.core.local_models import register_local_model
from factory.core.registry import Registry


def test_local_model_registration_and_offline_selection(data_dir, tiny_registered, monkeypatch, cls_spec):
    from factory.core.contracts import Constraints
    cfg = AppConfig(data_dir=data_dir, offline=True, models_dir=tiny_registered.parent)
    monkeypatch.setattr("factory.core.registry.load_catalog", lambda *a: pytest.fail("HF catalog used offline"))
    model = register_local_model(cfg, tiny_registered, languages=["ru"], license="mit", commercial_ok=True)
    assert model.source == "local"
    assert model.lora_modules["attention"]
    assert model.id == str(tiny_registered.resolve())
    registry = Registry.load(cfg)
    assert {m["id"] for m in registry.available_models(Constraints(), "ru")} == {model.id}
    assert Registry.load(cfg).model(model.id).source == "local"
    assert not registry.available_models(Constraints(), "en")


def test_registration_rejects_missing_or_outside_model_dir(data_dir, tiny_registered, tmp_path):
    cfg = AppConfig(data_dir=data_dir, models_dir=tmp_path / "empty")
    with pytest.raises(ValueError, match="должна находиться"):
        register_local_model(cfg, tiny_registered, languages=["ru"])
    cfg.models_dir = tiny_registered.parent
    with pytest.raises(FileNotFoundError):
        register_local_model(cfg, tmp_path / "missing", languages=["ru"])


def test_offline_search_never_uses_hub(cfg, registry, cls_spec, monkeypatch):
    from factory.search import funnel
    cfg.offline = True
    monkeypatch.setattr(funnel, "expand_query", lambda *a: pytest.fail("query expansion used offline"))
    monkeypatch.setattr(funnel, "load_catalog", lambda *a: pytest.fail("catalog used offline"))
    assert asyncio.run(funnel.search(cls_spec, registry, cfg)) == []


def test_offline_catalog_and_hf_dataset_are_disabled(cfg):
    from factory.search.catalog import load_catalog
    from factory.prep.loading import hf_full
    from unittest.mock import patch
    cfg.offline = True
    assert load_catalog("ru", cfg).records == []
    with patch("factory.core.config.get_config", return_value=cfg):
        with pytest.raises(RuntimeError, match="закрытом контуре"):
            hf_full("unknown/model")


def test_offline_mode_uses_uploaded_dataset(cfg, registry, cls_spec, monkeypatch):
    from factory.orchestrator import Orchestrator
    from factory.core.state import Step
    cfg.offline = True
    orch = Orchestrator(cfg, registry)
    st = orch.create_session()
    st.spec, st.step = cls_spec, Step.SPEC_CONFIRM
    orch.save(st)
    called = []

    async def no_background(sid, label, fn):
        called.append(label)

    monkeypatch.setattr(orch, "_background", no_background)
    monkeypatch.setattr(orch, "hardware", lambda: None)
    monkeypatch.setattr(orch.registry, "available_models", lambda *a: [{"id": cls_spec.base_model_id}])
    asyncio.run(orch._confirm_spec(st.session_id, None))
    assert called == ["ожидание датасета"]


def test_merge_stub_reports_unimplemented():
    from factory.prep.merge import MergeRequest, merge_datasets
    with pytest.raises(NotImplementedError, match="пока не реализовано"):
        merge_datasets(MergeRequest(sources=[Path("a.csv"), Path("b.csv")]))


def test_compose_exposes_persistent_data_and_readonly_models():
    import yaml
    root = Path(__file__).resolve().parents[1]
    service = yaml.safe_load((root / "docker-compose.yml").read_text(encoding="utf-8"))["services"]["factory"]
    assert "./data:/app/data" in service["volumes"]
    assert "./models:/models:ro" in service["volumes"]
    assert service["environment"]["FACTORY_OFFLINE"]


def test_api_registers_local_model_and_reports_merge_stub(data_dir, tiny_registered, monkeypatch):
    from fastapi.testclient import TestClient
    from factory.api.app import create_app
    from factory.orchestrator import Orchestrator
    cfg = AppConfig(data_dir=data_dir, offline=True, models_dir=tiny_registered.parent)
    orch = Orchestrator(cfg, Registry.load(cfg))
    client = TestClient(create_app(orch))
    result = client.post("/api/models/local", json={"path": str(tiny_registered), "languages": ["ru"]})
    assert result.status_code == 200, result.text
    assert result.json()["source"] == "local"
    assert client.get("/api/models/local").json()[0]["id"] == str(tiny_registered.resolve())
    assert client.post("/api/models/local", json={"path": str(data_dir), "languages": ["ru"]}).status_code == 400
    from factory.hparams import resources
    monkeypatch.setattr(orch, "hardware", lambda: type("Hardware", (), {"model_dump": lambda self: {}})())
    monkeypatch.setattr(resources, "rank_models", lambda models, hw: [m.id for m in models])
    assert client.get("/api/hardware").json()["models"] == [str(tiny_registered.resolve())]
    assert client.post("/api/datasets/merge", json={"sources": ["a.csv", "b.csv"]}).status_code == 501


def test_offline_requires_local_bert_score_before_training(cfg, registry, gen_spec, monkeypatch):
    from factory.orchestrator import Orchestrator
    from factory.core.state import Step
    cfg.offline = True
    orch = Orchestrator(cfg, registry)
    st = orch.create_session()
    st.spec, st.step = gen_spec, Step.SPEC_CONFIRM
    orch.save(st)
    monkeypatch.setattr(orch, "hardware", lambda: None)
    monkeypatch.setattr(orch.registry, "available_models", lambda *a: [{"id": gen_spec.base_model_id}])
    with pytest.raises(ValueError, match="bert_score_model"):
        asyncio.run(orch._confirm_spec(st.session_id, None))
    assert orch.load(st.session_id).step == Step.SPEC_CONFIRM


def test_cli_initializes_empty_persistent_volume(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from factory.__main__ import main
    from factory.core.config import get_config
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "empty-volume"
    monkeypatch.setenv("FACTORY_DATA_DIR", str(data))
    monkeypatch.setenv("FACTORY_SEED_REGISTRY_DIR", str(root / "data" / "registry"))
    monkeypatch.setenv("FACTORY_SEED_DOCS_DIR", str(root / "data" / "docs"))
    monkeypatch.setattr(sys, "argv", ["factory", "register-model", "--path", "placeholder"])
    monkeypatch.setattr("factory.core.local_models.register_local_model",
                        lambda *a, **k: SimpleNamespace(id="placeholder", params_b=1.0))
    get_config.cache_clear()
    try:
        main()
        assert (data / "registry" / "tasks.yaml").is_file()
        assert (data / "registry" / "models.yaml").is_file()
        assert (data / "docs" / "hparams").is_dir()
    finally:
        get_config.cache_clear()
