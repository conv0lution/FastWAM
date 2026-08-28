#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

ROUND4A_PYTHON="${ROUND4A_PYTHON:-$(command -v python)}"
export ROUND4A_PYTHON

exec "${ROUND4A_PYTHON}" -m experiments.asre_diagnosis.round4a.run_round4a "$@"
