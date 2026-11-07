#!/usr/bin/env bash
# Use the original submitted image plus the corrected launcher; no image build.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  printf 'Usage: %s DATASET_ROOT NEW_OUTPUT_DIRECTORY [IMAGE]\n' "$0" >&2
  exit 2
fi

bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
dataset_root="$(realpath -- "$1")"
output_root="$(realpath -m -- "$2")"
image="${3:-sunfish-v2d-submission:20261103}"
expected_image_id='sha256:2f78529c9100d710c1fe58d1f11dfb4795839cc066805de630c33f2f9f883b15'

[[ -d "$dataset_root" ]] || { printf 'Dataset directory missing: %s\n' "$dataset_root" >&2; exit 2; }
[[ ! -e "$output_root" && ! -L "$output_root" ]] || { printf 'Choose a new output directory: %s\n' "$output_root" >&2; exit 2; }
[[ -f "$bundle_dir/sunfish_container.py" ]] || { printf 'Corrected launcher missing.\n' >&2; exit 2; }
command -v docker >/dev/null
actual_image_id="$(docker image inspect --format '{{.Id}}' "$image")"
[[ "$actual_image_id" == "$expected_image_id" ]] || {
  printf 'WARNING: image ID differs from the tested submission: %s (tested %s). Continuing with the selected image; replay provenance checks remain active.\n' "$actual_image_id" "$expected_image_id" >&2
}

mkdir -- "$output_root"
mkdir -- "$output_root/output"
docker image inspect "$image" > "$output_root/image.inspect.json"
sha256sum -- "$bundle_dir/sunfish_container.py" > "$output_root/launcher.sha256"

command=(docker run --rm --pull=never --gpus all --network none --read-only
  --tmpfs /tmp --tmpfs /root --tmpfs /run
  --tmpfs /opt/runtime-crlf/robotic_grounding/flash_chord/src/flash_chord/assets/.cache
  --mount "type=bind,src=$dataset_root,dst=/data,readonly"
  --mount "type=bind,src=$output_root/output,dst=/output"
  --mount "type=bind,src=$bundle_dir/sunfish_container.py,dst=/submission/sunfish_container.py,readonly")
for name in EVAL_PYTHON KIT_PYTHON; do
  if [[ -n "${!name:-}" ]]; then
    command+=(--env "$name")
  fi
done
command+=("$image" reproduce --label tier1 --dataset-root /data --output /output --episodes 0)
printf '%q ' "${command[@]}" > "$output_root/command.sh"
printf '\n' >> "$output_root/command.sh"

set +e
"${command[@]}" > "$output_root/stdout.log" 2> "$output_root/stderr.log"
status=$?
set -e
printf '%s\n' "$status" > "$output_root/exitcode"
printf 'Exit code: %s\nComplete logs and replay outputs: %s\n' "$status" "$output_root"
exit "$status"
