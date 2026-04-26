FROM python:3.11-slim

LABEL maintainer="AgentQuant Team"
LABEL description="AgentQuant — Autonomous Quantitative Research Platform"

WORKDIR /app

# Install OS-level dependencies (for scipy/numpy native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency spec first (layer caching)
COPY pyproject.toml ./

# Install core + llm + data optional groups
RUN pip install --no-cache-dir -e ".[llm,data]"

# Copy source code
COPY . .

# Ensure data_store and experiments directories exist
RUN mkdir -p data_store experiments figures

# Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import src.utils.config" || exit 1

# Default: run the Streamlit dashboard
# Override CMD to run the agent instead: docker run agentquant python -m src.agent.runner
CMD ["streamlit", "run", "src/app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
