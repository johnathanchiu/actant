"""FastAPI assembly. Thin — config loading, CORS, lifespan, routes.

All the interesting work lives in app.coordinator + app.routes.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env BEFORE any module-level env reads. Demo server walks up
# from cwd to find the repo-root .env so API keys land in env.
load_dotenv(find_dotenv(usecwd=True))

from app.coordinator import build_coordinator  # noqa: E402
from app.routes import router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    coord = await build_coordinator()
    app.state.coordinator = coord
    try:
        yield
    finally:
        await coord.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="Actant Demo Server", lifespan=lifespan)

    # CORS: allow any localhost origin so the dev UI (on whatever
    # vite port) can talk to us. Set ACTANT_CORS_ORIGINS to override.
    explicit = os.getenv("ACTANT_CORS_ORIGINS", "").strip()
    if explicit:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in explicit.split(",") if o.strip()],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
