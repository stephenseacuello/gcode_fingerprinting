# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (libusb for DAQ libs, build tools for some wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential git libusb-1.0-0-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and source
COPY pyproject.toml pyproject.toml
COPY requirements.txt requirements.txt
COPY src/ src/
COPY . .

# Upgrade pip and install the project. Some environments (mac host / MCC
# vendor SDK) provide uldaq outside PyPI; we allow skipping uldaq via build-arg
RUN python -m pip install --upgrade pip setuptools wheel

ARG SKIP_ULDAQ=0
RUN if [ "$SKIP_ULDAQ" = "1" ]; then \
      python -m pip install --no-deps . && \
      grep -E -v '^\s*uldaq' requirements.txt | xargs -r python -m pip install ; \
    else \
      python -m pip install . ; \
    fi

# Expose common ports used by the project (Flask, Dash, ZMQ pub)
EXPOSE 5000 8050 5557

# By default drop to a shell so the container is interactive for development.
CMD ["bash"]
