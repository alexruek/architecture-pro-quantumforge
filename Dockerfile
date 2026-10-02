FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf_cache

# cron нужен контейнеру updater; бот его не использует
RUN apt-get update \
    && apt-get install -y --no-install-recommends cron \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-сборка torch, чтобы образ не тянул CUDA-библиотеки
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY deploy ./deploy
COPY golden_questions.txt ./
RUN chmod +x scripts/*.sh deploy/*.sh

EXPOSE 8000

# по умолчанию запускается бот, updater переопределяет команду в docker-compose.yml
CMD ["uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
