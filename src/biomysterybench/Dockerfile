# Per-problem image: the shared tools base plus this problem's data, already
# extracted under data/. Built once per problem by build_and_push.sh.
#
#   docker build -f Dockerfile --build-arg BASE=<base-image> \
#     -t <prefix><id> <dir-holding-this-problem's-data>
#
# build_and_push.sh always passes BASE, set to BP_BASE_TAG (default <BP_PREFIX>base).
# The default below only applies to a manual build without --build-arg, and then
# resolves to base:latest. Tag the image built from Dockerfile.base as base:latest
# locally first, or Docker tries to pull base:latest from Docker Hub.
ARG BASE=base
FROM ${BASE}

# The build context is the problem's own extracted data directory.
COPY . /workspace/data/

WORKDIR /workspace
