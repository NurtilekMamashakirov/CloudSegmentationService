FROM python:3.12

WORKDIR /fastApiProject

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root

COPY . .

WORKDIR /app

EXPOSE 8000

CMD ["poetry", "run", "gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "main:app"]