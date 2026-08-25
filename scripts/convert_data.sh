#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ARGS=()
while (($#)); do
  case "$1" in
    --backend)
      [[ $# -ge 2 ]] || { echo "--backend requires isaac_sim" >&2; exit 2; }
      [[ "$2" == "isaac_sim" || "$2" == "sim" ]] || { echo "real-world backend has been removed; use isaac_sim" >&2; exit 2; }
      shift 2
      ;;
    --backend=*)
      [[ "${1#*=}" == "isaac_sim" || "${1#*=}" == "sim" ]] || { echo "real-world backend has been removed; use isaac_sim" >&2; exit 2; }
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m flowcarrycbf.cli.convert_data "${ARGS[@]}"
