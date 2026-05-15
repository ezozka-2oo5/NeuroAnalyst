#!/usr/bin/env bash
# NeuroAnalyst FastAPI Backend ishga tushirish skripti
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# .env mavjudligini tekshirish
if [ ! -f ".env" ]; then
    echo "⚠️  .env fayli topilmadi. .env.example dan nusxalash..."
    cp .env.example .env
    echo "✅ .env yaratildi. SECRET_KEY va AES_MASTER_KEY ni o'zgartirishni unutmang!"
fi

# Virtual env
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# ONNX model mavjudligini tekshirish
ONNX_PATH="ml_pipeline/real_output/checkpoints/eeg_model.onnx"
DEMO_ONNX="ml_pipeline/demo_output/checkpoints/eeg_model.onnx"
if [ ! -f "$ONNX_PATH" ] && [ ! -f "$DEMO_ONNX" ]; then
    echo "⚠️  ONNX model topilmadi. Eksport qilinmoqda..."
    python3 -m ml_pipeline.export_onnx
fi

# Foydalanuvchi migratsiyasi (agar users jadvali bo'sh bo'lsa)
python3 -c "
import sys; sys.path.insert(0, '.')
from backend.models.db import create_all_tables, SessionLocal, get_db
from backend.models.db import User
create_all_tables()
db = SessionLocal()
if db.query(User).count() == 0:
    print('Foydalanuvchilar migratsiya qilinmoqda...')
    import subprocess
    subprocess.run(['python3', '-m', 'backend.auth.user_manager', '--migrate'])
db.close()
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🚀 NeuroAnalyst FastAPI Backend"
echo "  📍 Manzil: http://localhost:8000"
echo "  📖 Swagger: http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 -m uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
