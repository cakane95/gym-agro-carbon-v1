FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=2.3.2 python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

CMD ["tail", "-f", "/dev/null"]