FROM --platform=$TARGETPLATFORM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    libgmp-dev libmpfr-dev libmpc-dev \
    curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Install fpylll from PyPI (manylinux wheel on amd64). If a build is needed,
# deps above are enough for typical installs.
RUN pip install --no-cache-dir -U pip setuptools wheel \
  && pip install --no-cache-dir cysignals==1.12.4 \
  && pip install --no-cache-dir fpylll==0.6.4

COPY . /work

# Make sure strategies exist for wheels that hardcode a missing default path.
RUN test -f /work/fplll_strategies/default.json || true

CMD ["python3", "solve_svp.py"]
