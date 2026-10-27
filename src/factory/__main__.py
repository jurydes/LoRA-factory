"""CLI: python -m factory serve | build-catalog --lang ru."""
from __future__ import annotations

import argparse
import logging
import os

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
    cat = sub.add_parser("build-catalog", help="офлайн: кэш каталога HF и эмбеддинги описаний")
    cat.add_argument("--lang", default="ru")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if a.cmd == "serve":
        import uvicorn
        from factory.api.app import create_app
        uvicorn.run(create_app(), host=a.host, port=a.port)
    else:
        from factory.search.catalog import build_catalog
        build_catalog(a.lang)


if __name__ == "__main__":
    main()
