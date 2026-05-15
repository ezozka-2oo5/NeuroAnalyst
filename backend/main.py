"""
backend/main.py
=================
NeuroAnalyst FastAPI Backend — Asosiy ilova fayli.

Arxitektura:
  - FastAPI + Uvicorn ASGI server
  - JWT autentifikatsiya (barcha /api/* endpointlar uchun)
  - CORS — Streamlit frontend bilan ishlash uchun
  - GZip compression — katta EEG ma'lumotlarini tezroq yuborish
  - Lifespan — startup da model yuklab olish
  - Swagger UI: http://localhost:8000/docs
  - ReDoc: http://localhost:8000/redoc

Ishga tushirish:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

# Loyiha root'ini Python path'ga qo'shamiz
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.api.v1.routes.admin import router as admin_router
from backend.api.v1.routes.auth import router as auth_router
from backend.api.v1.routes.inference import router as inference_router
from backend.api.v1.routes.patients import router as patients_router
from backend.config import get_settings
from backend.inference.onnx_runner import initialize_model
from backend.models.db import create_all_tables

settings = get_settings()


# ── Lifespan (startup va shutdown) ────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Server ishga tushganda:
      1. Barcha DB jadvallarini yaratadi.
      2. ONNX/PyTorch modelini yuklab oladi (singleton).

    Server o'chirilganda:
      - Ochiq resurslarni yopadi.
    """
    print("━" * 60)
    print("  🧠 NeuroAnalyst Backend ishga tushmoqda...")
    print("━" * 60)

    # 1. Ma'lumotlar bazasi
    print("[1/2] Ma'lumotlar bazasi jadvallar tekshirilmoqda...")
    create_all_tables()
    print("      ✅ DB tayyorlandi.")

    # 2. Model yuklash
    print("[2/2] ML model yuklanmoqda...")
    model_type = initialize_model()
    if model_type == "none":
        print("      ⚠️  Model yuklanmadi — inference funksiyasi o'chirilgan.")
    else:
        print(f"      ✅ Model tayyor: {model_type.upper()}")

    print("━" * 60)
    print(f"  🚀 Server tayyor: http://{settings.backend_host}:{settings.backend_port}")
    print(f"  📖 Swagger UI:   http://localhost:{settings.backend_port}/docs")
    print("━" * 60)

    yield  # Ilova ishlaydi

    print("\n  🔴 NeuroAnalyst Backend o'chirilmoqda...")


# ── FastAPI ilovasi ────────────────────────────────────────────────────────


app = FastAPI(
    title="NeuroAnalyst Clinical API",
    description="""
## NeuroAnalyst EEG Tahlil Backend

Bu API quyidagi imkoniyatlarni taqdim etadi:

### 🔐 Autentifikatsiya
- **JWT Bearer Token** — barcha `/api/*` endpointlar himoyalangan.
- **Rollar:** `admin` (to'liq huquq), `doctor` (tahlil va bemorlar), `viewer` (faqat ko'rish).

### 🧠 EEG Inference
- ONNX Runtime (tezkor) yoki PyTorch (fallback) orqali real-time bashorat.
- Alzheimer, FTD va Sog'lom tasniflash yoki Seizure detection.

### 🔒 Xavfsizlik
- AES-256-GCM shifrlash — bemor EEG fayllari diskda shifrlangan.
- Audit log — barcha harakatlar qayd etiladi.
""",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── Middleware ─────────────────────────────────────────────────────────────


# GZip — katta JSON response lar uchun
app.add_middleware(GZipMiddleware, minimum_size=1024)

# CORS — Streamlit frontend bilan ishlash uchun
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Process-Time"],
)


# ── Request timing middleware ──────────────────────────────────────────────


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Har so'rov uchun bajarilish vaqtini headerga qo'shadi."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time * 1000:.2f}ms"
    return response


# ── Global exception handler ──────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Kutilmagan xatolarni JSON formatida qaytaradi."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Server ichki xatosi yuz berdi.",
            "type": type(exc).__name__,
        },
    )


# ── Router lar ────────────────────────────────────────────────────────────


app.include_router(auth_router)
app.include_router(inference_router)
app.include_router(patients_router)
app.include_router(admin_router)



# ── Asosiy endpointlar ────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "NeuroAnalyst Clinical API",
        "version": "2.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health", tags=["Tizim"], summary="Server sog'liq tekshiruvi")
def health_check():
    """Server ishlayotganini va model holatini tekshiradi."""
    from backend.inference.onnx_runner import get_model_status
    model_status = get_model_status()
    return {
        "status": "healthy",
        "model_ready": model_status["is_ready"],
        "model_type": model_status["model_type"],
        "timestamp": time.time(),
    }


# ── Ishga tushirish ────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
        log_level="info",
    )
