#!/bin/bash
set -e
SEARCH_DIR="$PWD/../../external/%LLVM_REPO_NAME%/bin"
REAL_BIN=$(find $SEARCH_DIR -path "*clang-tidy" -type f -print -quit 2>/dev/null)
if [ -z "$REAL_BIN" ]; then
  echo "Error: Failed to locate clang-tidy inside action workspace. Searched directory: $SEARCH_DIR" >&2
  exit 1
fi
echo "Using clang-tidy at: " $REAL_BIN
REAL_LIB_DIR="$(dirname "$REAL_BIN")/../lib"
export LD_LIBRARY_PATH="${REAL_LIB_DIR}:${LD_LIBRARY_PATH}"
# Intentional unquoted $@ expansion.
# Word-splitting is required here to resolve composite tokens passed by Bazel
# (e.g., "-include file.h" -> "-include" "file.h").
exec "$REAL_BIN" $@
