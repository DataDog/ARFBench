#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENTSLM_DIR="$REPO_ROOT/third_party/OpenTSLM"
OPENTSLM_SRC="$OPENTSLM_DIR/src"
TARGET_ARFBENCH_DIR="$OPENTSLM_SRC/time_series_datasets/arfbench"
CUSTOM_DIR="$REPO_ROOT/opentslm_custom"
REMOTE_URL="https://github.com/StanfordBDHG/OpenTSLM.git"

mkdir -p "$REPO_ROOT/third_party"

if [[ ! -d "$OPENTSLM_DIR/.git" ]]; then
  echo "Cloning OpenTSLM into $OPENTSLM_DIR"
  git clone "$REMOTE_URL" "$OPENTSLM_DIR"
else
  echo "OpenTSLM already exists at $OPENTSLM_DIR; skipping clone"
fi

if [[ ! -d "$TARGET_ARFBENCH_DIR" ]]; then
  echo "Creating missing ARFBench directory in OpenTSLM"
  mkdir -p "$TARGET_ARFBENCH_DIR"
fi

cp "$CUSTOM_DIR/ARFBenchQADataset.py" "$TARGET_ARFBENCH_DIR/ARFBenchQADataset.py"
cp "$CUSTOM_DIR/arfbench_loader.py" "$TARGET_ARFBENCH_DIR/arfbench_loader.py"

echo "Copied ARFBench OpenTSLM dataset files to:"
echo "  $TARGET_ARFBENCH_DIR"
echo
echo "OpenTSLM setup complete."
echo "Use --opentslm-src \"$OPENTSLM_SRC\" when running evaluation/eval_opentslm.py"
