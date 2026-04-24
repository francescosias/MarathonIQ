FROM python:3.10.6-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY project_logic project_logic
COPY api api
COPY models models
COPY setup.py setup.py

RUN pip install .

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
