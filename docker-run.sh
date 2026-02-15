#!/usr/bin/env bash
set -euo pipefail

img="zzzsosleepy-solver"

# Force x86_64. On Apple Silicon / arm64 hosts this uses emulation, but for this
# challenge it can be more stable than native arm64 fplll builds.
docker build --platform=linux/amd64 -t "$img" .

# Run solver in container (host networking not needed; outbound connect only).
docker run --rm --platform=linux/amd64 "$img"
