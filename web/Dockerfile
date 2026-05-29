FROM python:3.11-slim

WORKDIR /app

# gcc/g++ needed only if any dependency lacks a pre-built wheel
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    "git+https://github.com/yuanlinding/RepresentationAnalysis@43fa156a7d379e407ae500f80d5faee78c04bcdb"

COPY app.py .

RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
