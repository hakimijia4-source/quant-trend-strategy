#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DEMO=0
if [[ "${1:-}" == "--with-demo" ]]; then
    RUN_DEMO=1
elif [[ -n "${1:-}" ]]; then
    echo "Usage: bash scripts/bootstrap_wsl.sh [--with-demo]" >&2
    exit 2
fi

echo "Searching WSL for a Python 3.10+ environment containing PyTorch..."

candidate_file="$(mktemp)"
trap 'rm -f "${candidate_file}"' EXIT

# Prefer the currently activated environment, then inspect common Conda/venv
# locations. find is bounded to the user's home directory and never modifies it.
for command_name in python python3; do
    if command -v "${command_name}" >/dev/null 2>&1; then
        command -v "${command_name}" >>"${candidate_file}"
    fi
done

for root in \
    "${HOME}/miniconda3" \
    "${HOME}/anaconda3" \
    "${HOME}/miniforge3" \
    "${HOME}/mambaforge" \
    "${HOME}/.conda/envs" \
    "${HOME}/.virtualenvs"; do
    if [[ -d "${root}" ]]; then
        find "${root}" -maxdepth 4 \( -type f -o -type l \) \
            -path '*/bin/python*' 2>/dev/null >>"${candidate_file}" || true
    fi
done

# Catch custom venv locations without scanning mounted Windows drives.
find "${HOME}" -maxdepth 6 \( -type f -o -type l \) \
    -path '*/bin/python*' 2>/dev/null >>"${candidate_file}" || true

selected_python=""
fallback_python=""
while IFS= read -r candidate; do
    [[ -x "${candidate}" ]] || continue
    if ! "${candidate}" - <<'PY' >/dev/null 2>&1
import sys
import torch
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
    then
        continue
    fi

    info="$("${candidate}" - <<'PY'
import sys
import torch
print(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}")
PY
)"
    echo "Found: ${candidate} (${info})"

    if [[ -z "${fallback_python}" ]]; then
        fallback_python="${candidate}"
    fi
    if "${candidate}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
        >/dev/null 2>&1; then
        selected_python="${candidate}"
        break
    fi
done < <(awk '!seen[$0]++' "${candidate_file}")

if [[ -z "${selected_python}" ]]; then
    selected_python="${fallback_python}"
fi

if [[ -z "${selected_python}" ]]; then
    cat >&2 <<'EOF'
No compatible WSL Python environment containing PyTorch was found.
PyTorch may be installed in another WSL distribution or in Windows rather than WSL.
Run `wsl.exe --list --verbose` in PowerShell and open the distribution where it is installed.
EOF
    exit 1
fi

echo
echo "Selected: ${selected_python}"
PYTHON_BIN="${selected_python}" bash "${PROJECT_DIR}/scripts/setup_wsl.sh"

if [[ "${RUN_DEMO}" -eq 1 ]]; then
    echo
    echo "Starting the 180-session synthetic training demo..."
    cd "${PROJECT_DIR}"
    "${selected_python}" -m quant_trend --config config/demo.toml demo --sessions 180
fi

echo
echo "Bootstrap complete. Reuse this interpreter with:"
echo "  ${selected_python} -m quant_trend --config config/demo.toml demo --sessions 180"
