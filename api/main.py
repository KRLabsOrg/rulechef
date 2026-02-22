from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.config import settings
from api.routes import data, extraction, learning, project, rules
from api.state import sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Background task: clean up expired sessions every 5 minutes
    async def cleanup_loop():
        while True:
            await asyncio.sleep(300)
            removed = sessions.cleanup()
            if removed:
                print(f"Cleaned up {removed} expired session(s). Active: {sessions.count}")

    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()


app = FastAPI(title="RuleChef", version="0.1.0", lifespan=lifespan)

# CORS for dev (Vite on :5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(project.router)
app.include_router(data.router)
app.include_router(learning.router)
app.include_router(extraction.router)
app.include_router(rules.router)

# Serve built frontend in production
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    from fastapi.responses import FileResponse, JSONResponse

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
