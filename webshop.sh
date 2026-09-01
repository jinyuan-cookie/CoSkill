#!/usr/bin/env bash

set -Eeuo pipefail

# Set COSKILL_HTTP_PROXY when package downloads require a network proxy.
if [[ -n "${COSKILL_HTTP_PROXY:-}" ]]; then
    export HTTP_PROXY="${COSKILL_HTTP_PROXY}"
    export http_proxy="${COSKILL_HTTP_PROXY}"
    export HTTPS_PROXY="${COSKILL_HTTP_PROXY}"
    export https_proxy="${COSKILL_HTTP_PROXY}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SPACY_VERSION_SPEC="spacy>=3.7.2,<3.8.0"
SPACY_SM_URL="https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
SPACY_LG_URL="https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

echo "Using Python: $(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip --version

echo "Installing MKL and CPU-only FAISS..."
"$PYTHON_BIN" -m pip install mkl faiss-cpu

echo "Installing the pip-packaged OpenJDK 11..."
"$PYTHON_BIN" -m pip install "jdk4py==11.0.13.1"

JAVA_HOME="$($PYTHON_BIN -c 'from jdk4py import JAVA_HOME; print(JAVA_HOME)')"
export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

echo "JAVA_HOME=$JAVA_HOME"
java -version

echo "Installing spaCy compatible with the 3.7.1 English models..."
"$PYTHON_BIN" -m pip install "$SPACY_VERSION_SPEC"

echo "Installing required spaCy model: en_core_web_sm..."
if ! "$PYTHON_BIN" -m pip install "$SPACY_SM_URL"; then
    echo "Direct wheel installation failed; trying the spaCy downloader..."
    "$PYTHON_BIN" -m spacy download en_core_web_sm
fi

echo "Installing optional spaCy model: en_core_web_lg..."
if ! "$PYTHON_BIN" -m pip install "$SPACY_LG_URL"; then
    echo "Warning: en_core_web_lg installation failed; continuing because it is optional." >&2
fi

echo "Verifying installations..."
"$PYTHON_BIN" - <<'PY'
import subprocess

import faiss
import spacy
from jdk4py import JAVA, JAVA_HOME

spacy.load("en_core_web_sm")

print(f"✓ faiss installed successfully: {getattr(faiss, '__version__', 'unknown')}")
print("✓ en_core_web_sm installed successfully")
print(f"✓ JAVA_HOME: {JAVA_HOME}")
subprocess.run([str(JAVA), "-version"], check=True)

try:
    spacy.load("en_core_web_lg")
except Exception:
    print("! en_core_web_lg is not installed, but it is optional")
else:
    print("✓ en_core_web_lg installed successfully")
PY

echo "Persisting JAVA_HOME and PATH for future Bash sessions..."
JAVA_ENV_FILE="$HOME/.jdk4py_env.sh"
BASHRC_FILE="$HOME/.bashrc"
BASHRC_SOURCE_LINE='[ -f "$HOME/.jdk4py_env.sh" ] && . "$HOME/.jdk4py_env.sh"'

{
    printf 'export JAVA_HOME=%q\n' "$JAVA_HOME"
    printf 'export PATH="$JAVA_HOME/bin:$PATH"\n'
} > "$JAVA_ENV_FILE"

touch "$BASHRC_FILE"
if ! grep -Fqx "$BASHRC_SOURCE_LINE" "$BASHRC_FILE"; then
    printf '\n%s\n' "$BASHRC_SOURCE_LINE" >> "$BASHRC_FILE"
fi

echo
echo "Installation completed successfully."
echo "Permanent Java environment file: $JAVA_ENV_FILE"
echo "Bash startup file updated: $BASHRC_FILE"
echo "The settings will load automatically in new terminals."
echo "To apply them to the current terminal immediately, run:"
echo "source \"$JAVA_ENV_FILE\""
