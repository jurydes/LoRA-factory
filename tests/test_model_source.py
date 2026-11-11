"""Switching the model source persists and changes the available models."""

from fastapi.testclient import TestClient

from factory.api.app import create_app
from factory.core.config import load_config
from factory.core.contracts import Constraints
from factory.core.registry import Registry
from factory.orchestrator import Orchestrator


def test_model_source_switches_without_restart_and_survives_restart(data_dir, cfg, monkeypatch):
    monkeypatch.setenv("FACTORY_OFFLINE", "1")
    cfg.offline = True
    orch = Orchestrator(cfg, Registry.load(cfg))
    client = TestClient(create_app(orch))

    assert client.get("/api/model-source").json()["source"] == "local"
    assert orch.registry.available_models(Constraints(), "ru") == []

    response = client.put("/api/model-source", json={"source": "hf"})
    assert response.status_code == 200, response.text
    assert client.get("/api/model-source").json()["source"] == "hf"
    assert orch.registry.available_models(Constraints(), "ru")
    assert load_config(data_dir).offline is False

    response = client.put("/api/model-source", json={"source": "local"})
    assert response.status_code == 200, response.text
    assert orch.registry.available_models(Constraints(), "ru") == []
    assert load_config(data_dir).offline is True


def test_model_source_rejects_invalid_value(data_dir, cfg):
    client = TestClient(create_app(Orchestrator(cfg, Registry.load(cfg))))
    assert client.put("/api/model-source", json={"source": "other"}).status_code == 422


def test_sources_are_exclusive(registry):
    model = next(iter(registry.models.values())).model_copy(update={"id": "/models/example", "source": "local"})
    registry.add_models([model])
    registry.offline = True
    assert {m.id for m in registry.filter_models(Constraints())} == {model.id}
    registry.offline = False
    assert all(m.source != "local" for m in registry.filter_models(Constraints()))
