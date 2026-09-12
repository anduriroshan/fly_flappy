#!/usr/bin/env bash
# Install a portable Blender on a headless GPU box (Vast.ai etc.).
# No root / apt needed — Blender ships as a self-contained tarball.
#
#   bash scripts/setup_blender.sh            # installs to ./third_party/blender
#   source <(grep BLENDER_BIN ...)           # or just use the printed path
#
# After this, render with Cycles (headless GPU, no display):
#   export BLENDER_BIN=$(cat .blender_bin)
#   python -m scripts.render_morphology --recording runs/morphology/activation.npz
set -euo pipefail

BLENDER_VERSION="${BLENDER_VERSION:-4.2.3}"
BLENDER_MAJOR="${BLENDER_VERSION%.*}"     # e.g. 4.2
DEST="${1:-third_party/blender}"

mkdir -p "$DEST"
url="https://download.blender.org/release/Blender${BLENDER_MAJOR}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
tarball="$DEST/blender.tar.xz"

if [ ! -x "$DEST/blender" ]; then
    echo "[setup-blender] downloading $url"
    curl -L -o "$tarball" "$url"
    echo "[setup-blender] extracting..."
    tar -xf "$tarball" -C "$DEST" --strip-components=1
    rm -f "$tarball"
fi

BIN="$(readlink -f "$DEST/blender")"
echo "$BIN" > .blender_bin
echo "[setup-blender] Blender ready: $BIN"
"$BIN" --version | head -1

cat <<EOF

[setup-blender] Blender is a self-contained GPU renderer here — Cycles needs
NO display. Verify the GPU is visible to Blender:

    $BIN -b --python-expr "import bpy; p=bpy.context.preferences.addons['cycles'].preferences; p.compute_device_type='OPTIX'; print([d.name for d in p.get_devices_for_type('OPTIX')])"

Then render:
    export BLENDER_BIN=$BIN
    python -m scripts.render_morphology \\
        --recording runs/morphology/activation.npz --dry-run   # fail-fast check
EOF
