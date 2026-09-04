FROM python:3.12-slim

# libgomp — рантайм OpenMP, нужен LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости (кэшируемый слой), потом код приложения
COPY pyproject.toml README.md ./
COPY drift_guardian ./drift_guardian
RUN pip install --no-cache-dir .

COPY app ./app
COPY scripts ./scripts
COPY .streamlit ./.streamlit

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
