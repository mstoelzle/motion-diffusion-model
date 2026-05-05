#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DEST_DIR="$REPO_ROOT/body_models/g1"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if [ -f "$DEST_DIR/g1_29dof.urdf" ] && [ -d "$DEST_DIR/meshes" ]; then
  echo "G1 asset already exists at $DEST_DIR"
  exit 0
fi

mkdir -p "$(dirname "$DEST_DIR")"

echo "Downloading Holosoma G1 asset into $DEST_DIR"
git clone --depth 1 --filter=blob:none --sparse https://github.com/amazon-far/holosoma.git "$TMP_DIR/holosoma"
cd "$TMP_DIR/holosoma"
git sparse-checkout set src/holosoma/holosoma/data/robots/g1

rm -rf "$DEST_DIR"
cp -a src/holosoma/holosoma/data/robots/g1 "$DEST_DIR"

echo "Downloaded G1 asset:"
echo "  $DEST_DIR/g1_29dof.urdf"
echo "  $DEST_DIR/g1_29dof.xml"
