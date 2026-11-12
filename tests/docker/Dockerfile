FROM rclone/rclone:1.74.2 AS rclone

FROM python:3.11-slim

COPY --from=rclone /usr/local/bin/rclone /usr/local/bin/rclone

RUN pip install --no-cache-dir \
    "bittensor>=10.3,<11" \
    "psycopg[binary]>=3.2,<3.3" \
    "httpx>=0.27,<0.29" \
    "numpy>=1.26,<3" \
    "boto3>=1.34,<2" \
    "huggingface-hub>=0.34,<2" \
    "tokenizers>=0.22,<0.23" \
    "PyNaCl>=1.5,<2" \
    "jsonschema>=4.22,<5" \
    "pytest>=9,<10"

WORKDIR /app
ENV PYTHONPATH=/app

CMD ["sh", "-c", "python -m pytest -q -p no:cacheprovider tests/unit && python -m pytest -q -p no:cacheprovider tests/integration"]
