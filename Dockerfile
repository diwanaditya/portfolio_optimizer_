# Multi-stage build: keeps the final image to only what's needed to run
# the API, not the full build toolchain (scipy/numpy compile faster from
# wheels, but this still keeps the runtime image lean by not carrying
# pip's cache or build artifacts into the final layer).

FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
# Only the API's runtime dependencies are installed here -- benchmarking
# (PyPortfolioOpt/riskfolio-lib) and RL (torch) are large, optional
# extras not needed to SERVE the API; installing them in the production
# image would roughly double its size for code paths a typical deployment
# never calls. Split them into requirements-api.txt if your deployment
# does need /optimize/* comparisons or RL agents live in production.
RUN pip install --no-cache-dir --user \
    numpy==2.4.6 pandas==3.0.5 scipy==1.17.1 hmmlearn==0.3.3 \
    fastapi==0.141.1 uvicorn==0.52.1 pydantic==2.13.4 pydantic-settings==2.15.0 \
    slowapi==0.1.10 PyJWT==2.7.0 gunicorn==26.1.0 stripe==15.4.0 \
    matplotlib==3.10.8

FROM python:3.12-slim

# Run as a non-root user -- never run a production web process as root.
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
COPY --from=builder /root/.local /home/appuser/.local
COPY portfolio_optimizer/ ./portfolio_optimizer/
COPY gunicorn_conf.py .

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["gunicorn", "portfolio_optimizer.api.service:app", "-c", "gunicorn_conf.py"]
