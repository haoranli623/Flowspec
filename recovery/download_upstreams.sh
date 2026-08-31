#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--execute]" >&2
  echo "Default is a non-mutating plan. --execute downloads exact pinned upstreams." >&2
}

execute=false
case "${1:-}" in
  "") ;;
  --execute) execute=true ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

: "${FLOWSPEC_LEROBOT_ROOT:=/path/to/lerobot}"
: "${FLOWSPEC_MODEL_ROOT:=/path/to/models/flowspec-vla}"
: "${FLOWSPEC_DATASET_ROOT:=/path/to/datasets/flowspec-vla}"

lerobot_commit="8b256a6c0d4769c3cc3e7e98f04940126398a391"
base_revision="c83c3163b8ca9b7e67c509fffd9121e66cb96205"
recipe_revision="31d453f7edd78c839a8bbc39744a292686daf0de"
smolvlm_revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467"
dataset_revision="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"

if ! $execute; then
  echo "Pinned reconstruction plan (nothing downloaded):"
  echo "  LeRobot -> $FLOWSPEC_LEROBOT_ROOT @ $lerobot_commit"
  echo "  lerobot/smolvla_base -> $FLOWSPEC_MODEL_ROOT/smolvla_base @ $base_revision"
  echo "  lerobot/smolvla_libero -> $FLOWSPEC_MODEL_ROOT/smolvla_libero @ $recipe_revision"
  echo "  HuggingFaceTB/SmolVLM2-500M-Video-Instruct -> $FLOWSPEC_MODEL_ROOT/smolvlm @ $smolvlm_revision"
  echo "  lerobot/libero -> $FLOWSPEC_DATASET_ROOT @ $dataset_revision"
  echo "Set paths from recovery/path_map.example.env and rerun with --execute."
  exit 0
fi

for value in "$FLOWSPEC_LEROBOT_ROOT" "$FLOWSPEC_MODEL_ROOT" "$FLOWSPEC_DATASET_ROOT"; do
  case "$value" in
    /path/to/*) echo "Refusing placeholder destination: $value" >&2; exit 2 ;;
  esac
done
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v huggingface-cli >/dev/null || { echo "huggingface-cli is required" >&2; exit 1; }

test ! -e "$FLOWSPEC_LEROBOT_ROOT" || { echo "Destination already exists: $FLOWSPEC_LEROBOT_ROOT" >&2; exit 1; }
git clone https://github.com/huggingface/lerobot.git "$FLOWSPEC_LEROBOT_ROOT"
git -C "$FLOWSPEC_LEROBOT_ROOT" checkout --detach "$lerobot_commit"

mkdir -p "$FLOWSPEC_MODEL_ROOT" "$FLOWSPEC_DATASET_ROOT"
huggingface-cli download lerobot/smolvla_base --revision "$base_revision" --local-dir "$FLOWSPEC_MODEL_ROOT/smolvla_base"
huggingface-cli download lerobot/smolvla_libero --revision "$recipe_revision" --local-dir "$FLOWSPEC_MODEL_ROOT/smolvla_libero"
huggingface-cli download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --revision "$smolvlm_revision" --local-dir "$FLOWSPEC_MODEL_ROOT/smolvlm"
huggingface-cli download lerobot/libero --repo-type dataset --revision "$dataset_revision" --local-dir "$FLOWSPEC_DATASET_ROOT"

test "$(git -C "$FLOWSPEC_LEROBOT_ROOT" rev-parse HEAD)" = "$lerobot_commit"
echo "Pinned upstream downloads completed. Record downloaded file hashes before scientific use."
