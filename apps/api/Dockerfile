ARG PYTHON_IMAGE=public.ecr.aws/docker/library/python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

ARG CAMCAT_PREBUILT_RUNTIME=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/apps/api

RUN if [ "$CAMCAT_PREBUILT_RUNTIME" != "true" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends ffmpeg curl libgomp1 fonts-noto-cjk \
      && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app
COPY pyproject.toml alembic.ini ./
ARG CAMCAT_INSTALL_DEV=false
RUN mkdir -p apps/api/camcat && touch apps/api/camcat/__init__.py
RUN if [ "$CAMCAT_PREBUILT_RUNTIME" = "true" ]; then \
      echo "Using prebuilt local dependency runtime"; \
    elif [ "$CAMCAT_INSTALL_DEV" = "true" ]; then \
      pip install --timeout 120 --retries 8 '.[dev]'; \
    else \
      pip install --timeout 120 --retries 8 .; \
    fi
COPY apps/api ./apps/api
COPY tests ./tests
COPY scripts ./scripts
COPY packages/contracts ./packages/contracts
COPY apps/web/src/generated ./apps/web/src/generated
COPY docs ./docs
COPY infra/fonts/99-camcat-cjk.conf /etc/fonts/conf.d/99-camcat-cjk.conf
RUN pip install --no-deps .

EXPOSE 8000
CMD ["uvicorn", "camcat.main:app", "--host", "0.0.0.0", "--port", "8000"]
