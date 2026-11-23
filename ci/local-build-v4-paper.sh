#!/usr/bin/env bash
# Build the v4 paper PDF locally, mirroring openjournals/openjournals-draft-action
# (used in .github/workflows/paper-v4.yml). Requires Docker.
# Output: docs/paper-v4/paper.pdf
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOURNAL="${JOURNAL:-joss}"

docker run --rm \
    --volume "${REPO_ROOT}:/data" \
    --workdir /data \
    --user "$(id -u):$(id -g)" \
    --env JOURNAL="${JOURNAL}" \
    --env GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)" \
    openjournals/inara:latest \
    -o pdf docs/paper-v4/paper.md

echo "Built ${REPO_ROOT}/docs/paper-v4/paper.pdf"
