# Production-oriented container for Streamlit dashboard.
# Mount model directories and Data/ at runtime; do not bake secrets into the image.

FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt requirements-dev.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

EXPOSE 8501

# Override at runtime: EMBEDDING_MODEL_LOCAL_PATH, HF_CHAT_MODEL_LOCAL_PATH, OFFLINE_ONLY
CMD ["streamlit", "run", "App/dashboard.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
