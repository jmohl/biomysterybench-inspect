#!/usr/bin/env bash
#
# Build and push the BioMysteryBench sandbox images: one shared tools base, then one
# per-problem image (base + that problem's data under data/). The eval names a sample's
# image <BP_PREFIX><id>, so pass the same BP_PREFIX as BMB_IMAGE_PREFIX at run time.
#
# Requires: docker (linux/amd64) and push creds for BP_PREFIX's registry
# (`docker login <registry>`). The full set also needs HF_TOKEN with access to the
# gated dataset. The preview set downloads without a token.
#
#   HF_TOKEN=hf_... \
#   BP_PREFIX=registry.example.com/biomystery: \
#   ./build_and_push.sh
#
# Options (env):
#   BP_SOURCE=full|preview   dataset (default full)
#   BP_IDS="hb002 hb020"     build only these problem ids (default: all)
#   BP_MAX_GB=15             skip problems whose archive exceeds this (default: no cap)
#   BP_PLATFORM=linux/amd64  build platform (default linux/amd64)
#   BP_BASE_TAG              base image tag (default <BP_PREFIX>base)
#   BP_SKIP_EXISTING=1       skip problems whose image is already in the registry
#   BP_PRUNE_BUILD_CACHE=1   run `docker builder prune -af` after each problem. This
#                            clears the whole host's build cache, not only this
#                            script's, so it is off by default.
#
# Each problem image is removed from the local image store once pushed. The build
# cache still keeps a copy of each problem's data unless BP_PRUNE_BUILD_CACHE=1.
set -euo pipefail

SOURCE="${BP_SOURCE:-full}"
case "$SOURCE" in
    full) : "${HF_TOKEN:?set HF_TOKEN (access to the gated full dataset is needed)}" ;;
    preview) ;;
    *) echo "BP_SOURCE must be 'full' or 'preview', got '$SOURCE'" >&2; exit 2 ;;
esac
: "${BP_PREFIX:?set BP_PREFIX to your image repo, e.g. registry.example.com/biomystery:}"
PLATFORM="${BP_PLATFORM:-linux/amd64}"
BASE_TAG="${BP_BASE_TAG:-${BP_PREFIX}base}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# The repo id and revision are read from the module, which is the single source of
# the pin. problems.csv is fetched at run time at that pin while data/<id>.zip is
# baked into images at build time here, and a mismatch between the two runs and
# grades normally against wrong question and data pairs, with nothing to catch it.
read -r REPO REV < <(
    cd "$REPO_ROOT" && uv run python -c "
from biomysterybench.biomysterybench import (
    FULL_REPO, FULL_REVISION, PREVIEW_REPO, PREVIEW_REVISION,
)
full = '$SOURCE' == 'full'
print(FULL_REPO if full else PREVIEW_REPO, FULL_REVISION if full else PREVIEW_REVISION)
"
)

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "==> Building base image $BASE_TAG"
docker build --platform "$PLATFORM" -f "$SCRIPT_DIR/Dockerfile.base" -t "$BASE_TAG" "$SCRIPT_DIR"
docker push "$BASE_TAG"

echo "==> Reading problem list from $REPO"
(cd "$REPO_ROOT" && uv run python - "$REPO" "$REV" "$work/ids.txt" \
    "$work/sizes.json" "$SOURCE") <<'PY'
import csv, json, os, sys
from huggingface_hub import HfApi, hf_hub_download

repo, rev, out, sizes_out, source = sys.argv[1:6]
token = os.environ.get("HF_TOKEN") or None
path = hf_hub_download(repo, "problems.csv", repo_type="dataset", revision=rev,
                       token=token)
with open(path, newline="", encoding="utf-8") as f, \
     open(out, "w", encoding="utf-8") as w:
    for row in csv.DictReader(f):
        w.write(row["id"] + "\n")

# dataset_info covers the whole repo, so it is called once here and the sizes are
# written out for the per-problem step to read.
sizes = {}
if source != "preview":
    info = HfApi().dataset_info(repo, revision=rev, files_metadata=True, token=token)
    sizes = {s.rfilename: (s.size or 0) for s in info.siblings or []}
with open(sizes_out, "w", encoding="utf-8") as w:
    json.dump(sizes, w)
PY

ids="$(tr '\n' ' ' < "$work/ids.txt")"
[ -n "${BP_IDS:-}" ] && ids="$BP_IDS"

for id in $ids; do
    image="${BP_PREFIX}${id}"

    if [ "${BP_SKIP_EXISTING:-0}" = "1" ] && docker manifest inspect "$image" >/dev/null 2>&1; then
        echo "==> $id: image exists, skipping"
        continue
    fi

    echo "==> $id: fetching data"
    ctx="$work/$id"; mkdir -p "$ctx"
    rc=0
    (cd "$REPO_ROOT" && uv run python - "$REPO" "$REV" "$id" "$SOURCE" "$ctx" \
        "${BP_MAX_GB:-0}" "$work/sizes.json") <<'PY' || rc=$?
import json, os, sys, zipfile

from huggingface_hub import hf_hub_download

repo, rev, pid, source, ctx, max_gb, sizes_path = sys.argv[1:8]
token = os.environ.get("HF_TOKEN") or None
arc = "data.zip" if source == "preview" else f"data/{pid}.zip"
if float(max_gb) > 0 and source != "preview":
    with open(sizes_path, encoding="utf-8") as f:
        sizes = json.load(f)
    if arc not in sizes:
        print(f"    no archive for {pid}", file=sys.stderr); sys.exit(3)
    if sizes[arc] / 1e9 > float(max_gb):
        print(f"    {pid} archive {sizes[arc]/1e9:.1f}GB > BP_MAX_GB={max_gb}, skip",
              file=sys.stderr); sys.exit(3)


def drop_cached(path):
    # hf_hub_download returns a symlink into the HF cache. Both the link and the
    # blob it points at are removed.
    blob = os.path.realpath(path)
    for target in (path, blob):
        if os.path.lexists(target):
            os.unlink(target)


path = hf_hub_download(repo, arc, repo_type="dataset", revision=rev, token=token)
if source == "preview":
    # The preview bundle stays cached between problems and between builds, so a
    # truncated or corrupt copy would fail every later build. It is fetched again
    # once. Other failures, such as a full disk, keep the cached copy.
    try:
        with zipfile.ZipFile(path) as z:
            corrupt = z.testzip() is not None
    except zipfile.BadZipFile:
        corrupt = True
    if corrupt:
        print("    cached preview bundle is corrupt, downloading it again",
              file=sys.stderr)
        drop_cached(path)
        path = hf_hub_download(repo, arc, repo_type="dataset", revision=rev,
                               token=token)
try:
    with zipfile.ZipFile(path) as z:
        if source == "preview":
            # bundled zip holds every problem under <id>/; extract only this one, flatten
            members = [m for m in z.namelist() if m.startswith(f"{pid}/")]
            if not members:
                print(f"    {pid} not in preview bundle", file=sys.stderr); sys.exit(3)
            for m in members:
                z.extract(m, ctx)
            inner = os.path.join(ctx, pid)
            for name in os.listdir(inner):
                os.rename(os.path.join(inner, name), os.path.join(ctx, name))
            os.rmdir(inner)
        else:
            z.extractall(ctx)
finally:
    # hf_hub_download caches each archive in the HF cache (~/.cache/huggingface,
    # or wherever HF_HOME or HF_HUB_CACHE point). A full-set archive belongs to one
    # problem and is removed after use, because the full set is 155 GB and would
    # otherwise exhaust the runner disk. The preview bundle is shared by every
    # problem and stays cached, so later problems and later builds reuse it.
    if source != "preview":
        drop_cached(path)
PY
    if [ "$rc" -ne 0 ]; then
        rm -rf "$ctx"
        # Exit 3 is a deliberate skip: the archive is over BP_MAX_GB or absent from
        # the bundle. Any other code is a real failure -- an HF 401/429, a network
        # drop, a full disk, a corrupt zip -- and aborts, because a problem with no
        # image surfaces mid-eval as a docker pull error.
        if [ "$rc" -eq 3 ]; then
            echo "    (skipped $id)"; continue
        fi
        echo "!!! $id: fetch failed with exit $rc, aborting" >&2
        exit "$rc"
    fi

    echo "==> $id: build + push $image"
    docker build --platform "$PLATFORM" -f "$SCRIPT_DIR/Dockerfile" \
        --build-arg "BASE=$BASE_TAG" -t "$image" "$ctx"
    docker push "$image"
    # Reclaim disk before the next problem, whose archive can be up to ~27 GB.
    rm -rf "$ctx"
    docker image rm "$image" >/dev/null
    if [ "${BP_PRUNE_BUILD_CACHE:-0}" = "1" ]; then
        docker builder prune -af >/dev/null
    fi
done

echo "==> Done. Run the eval with:"
echo "    BMB_IMAGE_PREFIX=$BP_PREFIX inspect eval biomysterybench/bio_mystery_bench -T source=$SOURCE"
