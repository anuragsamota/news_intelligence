FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[chroma]"

ENV NI_VECTOR_DB_PATH=/data/vectordb
VOLUME ["/data", "/reports"]
ENTRYPOINT ["news-intelligence"]
CMD ["status"]
