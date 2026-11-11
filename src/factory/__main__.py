"""CLI: serve, build-catalog, register-model."""
from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows без режима разработчика: кэш HF без симлинков


def main() -> None:
    ap = argparse.ArgumentParser(prog="factory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="веб-чат и API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    cat = sub.add_parser("build-catalog", help="предварительная сборка каталога датасетов HF")
    cat.add_argument("--lang", default="ru")
    local = sub.add_parser("register-model", help="зарегистрировать каталог локальной модели")
    local.add_argument("--path", required=True, type=Path)
    local.add_argument("--lang", nargs="+", default=["multi"])
    local.add_argument("--license", default="")
    local.add_argument("--commercial-ok", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from factory.core.config import get_config
    cfg = get_config()
    seed_dir = Path(os.environ.get("FACTORY_SEED_REGISTRY_DIR", str(Path(__file__).resolve().parents[2] / "data" / "registry")))
    cfg.registry_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tasks.yaml", "models.yaml"):
        dest = cfg.registry_dir / name
        if not dest.exists():
            shutil.copy2(seed_dir / name, dest)
    docs_seed = Path(os.environ.get("FACTORY_SEED_DOCS_DIR", str(Path(__file__).resolve().parents[2] / "data" / "docs")))
    if docs_seed.exists() and not (cfg.data_dir / "docs").exists():
        shutil.copytree(docs_seed, cfg.data_dir / "docs")

    if a.cmd == "serve":
        import uvicorn
        from factory.api.app import create_app
        uvicorn.run(create_app(), host=a.host, port=a.port)
    elif a.cmd == "register-model":
        from factory.core.local_models import register_local_model
        model = register_local_model(cfg, a.path, languages=a.lang, license=a.license,
                                     commercial_ok=a.commercial_ok)
        print(f"Зарегистрирована модель: {model.id} ({model.params_b:g} млрд параметров)")
    else:
        from factory.search.catalog import build_catalog
        build_catalog(a.lang)


if __name__ == "__main__":
    main()
