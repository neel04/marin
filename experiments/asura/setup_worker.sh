#!/usr/bin/env bash
set -euo pipefail

archive=${1:?pass the local Marin source archive path}
root="$HOME/marin-asura-dclm"
mkdir -p "$root/repo" "$root/wandb" "$HOME/.config/marin/clusters" "$HOME/.config/wandb"
tar -xzf "$archive" -C "$root/repo"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
cd "$root/repo"
uv sync --package marin-core --extra tpu --extra dedup --no-dev
.venv/bin/python -c 'import jax, marin, levanter; print("jax", jax.__version__, "devices", [str(x) for x in jax.devices()])'
