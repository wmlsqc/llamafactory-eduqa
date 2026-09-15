FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 eduqa
USER eduqa
EXPOSE 8080
CMD ["python", "-m", "eduqa", "serve", "--host", "0.0.0.0", "--port", "8080"]
