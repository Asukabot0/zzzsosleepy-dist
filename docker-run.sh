#!/usr/bin/env bash
set -euo pipefail

img="zzzsosleepy-solver"

docker build -t "$img" .

# Run solver in container (host networking not needed; outbound connect only).
docker run --rm "$img"
