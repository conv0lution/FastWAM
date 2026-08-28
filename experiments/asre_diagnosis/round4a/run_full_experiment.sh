#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

ROUND4A_PYTHON="${ROUND4A_PYTHON:-${FASTWAM_PYTHON:-/home/zhaizicheng/miniconda3/envs/fastwam/bin/python3.10}}"
export ROUND4A_PYTHON

exec "${ROUND4A_PYTHON}" -m experiments.asre_diagnosis.round4a.run_round4a "$@"
