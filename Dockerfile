FROM python:3.6-slim-stretch AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /source
COPY requirements.txt .

RUN apt-get update -y \
	&& python -m venv /opt/venv \
	&& pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt





FROM python:3.6-slim-stretch AS final

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv
WORKDIR /source
COPY . .

RUN apt-get update -y \
	&& apt-get install libgomp1

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000", "--root-path=/ml"]