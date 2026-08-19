#!/usr/bin/env bash
set -euo pipefail

# Run this script from the WSL Python/Conda environment that already contains
# PyTorch.  The source tree is registered with a .pth file, avoiding pip build
# isolation and any network request that could replace the CUDA-specific wheel.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Project: ${PROJECT_DIR}"
echo "Python:  $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

${PYTHON_BIN} - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10+ is required; found {sys.version.split()[0]}")

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is not visible in this Python environment. Activate the WSL "
        "Conda/venv environment that contains torch, then rerun this script."
    ) from exc

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA build: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

missing_modules="$(${PYTHON_BIN} - <<'PY'
import importlib.util

required = ("numpy", "pandas", "torch")
print(" ".join(name for name in required if importlib.util.find_spec(name) is None))
PY
)"

if [[ -n "${missing_modules}" ]]; then
    echo "Missing modules in ${PYTHON_BIN}: ${missing_modules}"
    echo "Installing missing binary packages with an extended network timeout..."
    # Only missing runtime modules are installed. PyTorch is normally present
    # before this point, so its CUDA wheel is not replaced. Binary-only avoids
    # slow or fragile source builds inside WSL.
    ${PYTHON_BIN} -m pip install \
        --only-binary=:all: \
        --disable-pip-version-check \
        --timeout 180 \
        --retries 8 \
        ${missing_modules}

    still_missing="$(${PYTHON_BIN} - <<'PY'
import importlib.util

required = ("numpy", "pandas", "torch")
print(" ".join(name for name in required if importlib.util.find_spec(name) is None))
PY
)"
    if [[ -n "${still_missing}" ]]; then
        echo "Modules are still unavailable: ${still_missing}" >&2
        exit 1
    fi
fi

${PYTHON_BIN} - "${PROJECT_DIR}/src" <<'PY'
from pathlib import Path
import site
import sys

source_dir = Path(sys.argv[1]).resolve()
candidates = [Path(path) for path in site.getsitepackages()]
candidates.append(Path(site.getusersitepackages()))
errors = []
for directory in candidates:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "quant_trend_strategy.pth"
        marker.write_text(str(source_dir) + "\n", encoding="utf-8")
        print(f"Registered source path: {marker}")
        break
    except OSError as exc:
        errors.append(f"{directory}: {exc}")
else:
    raise SystemExit("Could not register the project:\n" + "\n".join(errors))
PY

cd "${PROJECT_DIR}"
${PYTHON_BIN} - <<'PY'
import numpy
import pandas
import torch
import quant_trend

print("Imports OK")
print(f"NumPy: {numpy.__version__}")
print(f"pandas: {pandas.__version__}")
print(f"quant_trend: {quant_trend.__file__}")
PY

${PYTHON_BIN} -m unittest discover -s tests -v

echo
echo "WSL environment is ready. Start with:"
echo "  ${PYTHON_BIN} -m quant_trend --config config/demo.toml demo --sessions 180"
