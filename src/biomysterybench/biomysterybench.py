"""
BioMysteryBench: open-ended agentic bioinformatics analysis.

Anthropic
https://huggingface.co/datasets/Anthropic/BioMysteryBench-full
https://huggingface.co/datasets/Anthropic/BioMysteryBench-preview

Each problem gives the model a real, anonymized biological dataset and asks a
question that requires actual analysis (alignment, expression, variant calling,
motif discovery, structure, ...) to answer. The model works in a sandbox with bio
tools and internet access, then submits a free-form answer that a model judge
grades against the problem's rubric.

# open 5-problem preview set, no token needed
inspect eval biomysterybench/bio_mystery_bench -T source=preview

# full 90-problem gated set, 5 trials
inspect eval biomysterybench/bio_mystery_bench -T source=full -T epochs=5

By default each problem's data is baked into a per-problem sandbox image (built
by build_and_push.sh); a sample's image is named <image_prefix><id> and reaches
the sandbox via Sample.metadata. With data_source="hf" or "s3", every sample
uses one shared base image instead and its data is downloaded into the sandbox
when the sample starts (see staging.py). See README.md.
"""

import csv
import json
import logging
import os
import re
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.solver import Solver

from biomysterybench.manifest import TIERS, ProblemArchive, load_source
from biomysterybench.scorer import biomystery_grader
from biomysterybench.solver import biomystery_solver
from biomysterybench.staging import DATA_SOURCES, stage_problem_data

logger = logging.getLogger(__name__)

FULL_REPO = "Anthropic/BioMysteryBench-full"
PREVIEW_REPO = "Anthropic/BioMysteryBench-preview"
FULL_REVISION = "b5a889c4757214ec9a6ade876b734f920a7799db"
PREVIEW_REVISION = "51c9024021b8989a0cb06ae623b02f90d14c2da3"

_COMPOSE = str(Path(__file__).parent / "compose.yaml")
# compose.yaml plus an ephemeral-storage request and limit for Kubernetes.
_COMPOSE_DISK = str(Path(__file__).parent / "compose_disk.yaml")

# A sample's image tag is this prefix + the problem id. Set the prefix (your image
# repo, e.g. registry.example.com/biomystery:) at run time via BMB_IMAGE_PREFIX or
# the image_prefix arg.
_IMAGE_PREFIX_ENV = "BMB_IMAGE_PREFIX"
# The shared base image for data_source="hf" or "s3". Pin it by digest.
_BASE_IMAGE_ENV = "BMB_BASE_IMAGE"

_DEFAULT_SANDBOX_MEMORY = "32G"
# Docker compose byte values accepted by the k8s_sandbox compose converter.
_MEMORY_RE = re.compile(r"^\d+(\.\d+)?(gb?|mb?|kb?|b)$", re.IGNORECASE)
# Kubernetes quantities for ephemeral-storage, e.g. "50Gi".
_DISK_RE = re.compile(r"^\d+(\.\d+)?(Ki|Mi|Gi|Ti|k|M|G|T)?$")

_YES = {"yes", "true", "1", "y"}

# "<comparability>-<interface>": the number changes when results stop being
# comparable with earlier runs, and the letter changes when task arguments change
# incompatibly. See the changelog in README.md.
TASK_VERSION = "2-B"


def _load_problems(repo: str, revision: str, token: str | None) -> list[dict[str, str]]:
    """Download problems.csv (pinned) from HF and return one dict per problem row."""
    csv_path = hf_hub_download(
        repo, "problems.csv", repo_type="dataset", revision=revision, token=token
    )
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _make_sample(
    rec: dict[str, str],
    image_prefix: str,
    sandbox_memory: str = _DEFAULT_SANDBOX_MEMORY,
    *,
    image: str | None = None,
    sandbox_disk: str | None = None,
    archive: ProblemArchive | None = None,
) -> Sample:
    """Map a problems.csv row to a Sample.

    `image` overrides the per-problem image (the shared base image when the data
    is staged at run time).
    """
    meta: dict[str, Any] = {
        "problem_id": rec["id"],
        "human_solvable": rec.get("human_solvable"),
        "allowed_domains": rec.get("allowed_domains"),
        # Substituted into compose.yaml as ${SAMPLE_METADATA_BMB_IMAGE}.
        "BMB_IMAGE": image or f"{image_prefix}{rec['id']}",
        # Substituted into compose.yaml as ${SAMPLE_METADATA_BMB_MEMORY}.
        "BMB_MEMORY": sandbox_memory,
    }
    if sandbox_disk is not None:
        # Substituted into compose_disk.yaml as ${SAMPLE_METADATA_BMB_DISK}.
        meta["BMB_DISK"] = sandbox_disk
    if archive is not None:
        meta["tier"] = archive.tier
        meta["archive_sha256"] = archive.sha256
    return Sample(
        input=rec["question"],
        target=rec["answer_rubric"],
        id=rec["id"],
        metadata=meta,
    )


def _harness_commit() -> str | None:
    """The git commit this package was installed from, if pip/uv recorded one."""
    try:
        raw = importlib_metadata.distribution("biomysterybench").read_text(
            "direct_url.json"
        )
    except importlib_metadata.PackageNotFoundError:
        return None
    commit = json.loads(raw or "{}").get("vcs_info", {}).get("commit_id")
    return str(commit) if commit else None


def _create_task(
    dataset: Dataset,
    solver: Solver | list[Solver],
    *,
    sandbox: str | tuple[str, str] | None = None,
    epochs: int = 1,
    metadata: dict[str, Any] | None = None,
) -> Task:
    """Assemble the Task (shared with tests, which inject a MemoryDataset)."""
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=biomystery_grader(telemetry=sandbox is not None),
        epochs=Epochs(epochs, "mean"),
        sandbox=sandbox,
        version=TASK_VERSION,
        metadata=metadata,
    )


@task
def bio_mystery_bench(
    source: str = "full",
    ids: str | list[str] | None = None,
    limit: int | None = None,
    human_solvable_only: bool = False,
    tier: str | list[str] | None = None,
    data_source: str = "image",
    image_prefix: str | None = None,
    base_image: str | None = None,
    s3_uri: str | None = None,
    sandbox_memory: str = _DEFAULT_SANDBOX_MEMORY,
    sandbox_disk: str | None = None,
    stage_timeout: int = 7200,
    with_sandbox: bool = True,
    epochs: int = 1,
    tool_timeout: int = 600,
    message_limit: int | None = 150,
    token_limit: int | None = None,
    max_tool_output: int | None = None,
) -> Task:
    """BioMysteryBench as an Inspect task.

    The model finds the problem's data extracted under `data/`. By default it is
    baked into a per-problem sandbox image. With data_source "hf" or "s3", one
    shared base image is used and the data is downloaded into the sandbox when
    the sample starts, laid out exactly as in the per-problem image.

    Args:
        source: "full" (90 gated problems) or "preview" (open 5-problem sample).
        ids: restrict to these problem ids. Accepts a list or a single id.
        limit: keep only the first N problems (after filtering).
        human_solvable_only: keep only problems >=1 human solved.
        tier: keep only problems in these archive-size tiers ("small" < 2 GB,
            "medium" < 10 GB, "large"), read from manifest.json. Accepts a list
            or a single tier.
        data_source: where the sandbox gets the problem data. "image" (default)
            uses the per-problem image. "hf" downloads the archive from a signed
            HuggingFace link resolved with the runner's HF_TOKEN. "s3" downloads
            it from a presigned URL into an S3 mirror (see s3_uri). The token and
            AWS credentials never enter the sandbox.
        image_prefix: per-problem image tag prefix; a sample's image is
            image_prefix + problem_id. Defaults to the BMB_IMAGE_PREFIX env var.
            Used only with data_source="image".
        base_image: shared image for data_source "hf" or "s3", ideally pinned by
            digest. Defaults to the BMB_BASE_IMAGE env var.
        s3_uri: s3://bucket/prefix of the mirror for data_source="s3". Defaults
            to the BMB_S3_URI env var.
        sandbox_memory: memory for each sandbox as a docker compose byte value
            (e.g. "32G", "512M"). On Kubernetes this is both the memory request and
            the memory limit of the sandbox pod.
        sandbox_disk: ephemeral storage for each sandbox as a Kubernetes quantity
            (e.g. "50Gi"), set as both the request and the limit of the pod.
            Docker ignores it. None (default) sets no disk request.
        stage_timeout: seconds allowed to download and extract a problem's data
            (data_source "hf" or "s3").
        with_sandbox: attach the docker sandbox (default True). False builds the task
            without a sandbox, for registration and offline tests.
        epochs: trials per problem.
        tool_timeout: per bash/python call timeout in seconds (passed to the solver).
        message_limit: hard cap on messages per sample
        token_limit: optional per-sample token cap (passed to the solver).
        max_tool_output: truncate each tool result to this many bytes (passed to the
            solver).

    """
    if source not in ("full", "preview"):
        raise ValueError(f"source must be 'full' or 'preview', got {source!r}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be zero or more, got {limit}")
    if not _MEMORY_RE.match(sandbox_memory):
        raise ValueError(
            "sandbox_memory must be a docker compose byte value such as '32G' or "
            f"'512M', got {sandbox_memory!r}"
        )
    if sandbox_disk is not None and not _DISK_RE.match(sandbox_disk):
        raise ValueError(
            "sandbox_disk must be a Kubernetes quantity such as '50Gi', got "
            f"{sandbox_disk!r}"
        )
    if data_source not in DATA_SOURCES:
        raise ValueError(
            f"data_source must be one of {DATA_SOURCES}, got {data_source!r}"
        )
    tiers = [tier] if isinstance(tier, str) else list(tier or [])
    if set(tiers) - set(TIERS):
        raise ValueError(
            f"tier must be among {TIERS}, got {sorted(set(tiers) - set(TIERS))}"
        )
    staged = data_source != "image"

    prefix = (
        image_prefix if image_prefix is not None else os.environ.get(_IMAGE_PREFIX_ENV)
    )
    base = base_image if base_image is not None else os.environ.get(_BASE_IMAGE_ENV)
    if with_sandbox and not staged and not prefix:
        raise ValueError(
            "image_prefix is unset; pass -T image_prefix=... or set BMB_IMAGE_PREFIX "
            "to your per-problem image repo (see README.md)"
        )
    if with_sandbox and staged and not base:
        raise ValueError(
            f"base_image is unset; data_source={data_source!r} needs the shared base "
            "image (pass -T base_image=... or set BMB_BASE_IMAGE)"
        )
    if staged and base and "@sha256:" not in base:
        logger.warning(
            "base_image %s is not pinned by digest; a rebuilt image under the same "
            "tag could change the tools between runs",
            base,
        )

    token = os.environ.get("HF_TOKEN") or None
    repo = FULL_REPO if source == "full" else PREVIEW_REPO
    revision = FULL_REVISION if source == "full" else PREVIEW_REVISION

    rows = _load_problems(repo, revision, token)
    archives = load_source(source, repo, revision)
    unknown = {r["id"] for r in rows} - set(archives)
    if unknown:
        raise ValueError(
            f"problems.csv has ids missing from manifest.json: {sorted(unknown)}; "
            "regenerate the manifest with scripts/make_manifest.py"
        )
    if tiers:
        rows = [r for r in rows if archives[r["id"]].tier in tiers]
    if human_solvable_only:
        rows = [
            r for r in rows if str(r.get("human_solvable", "")).strip().lower() in _YES
        ]
    if ids:
        # Inspect's CLI collapses a single-element list back to a bare string, so
        # `-T ids=hb002` arrives as "hb002". set() over a bare string is a set of
        # characters, so it is wrapped first.
        ids = [ids] if isinstance(ids, str) else ids
        want = set(ids)
        missing = want - {r["id"] for r in rows}
        if missing:
            raise ValueError(f"ids not in the dataset: {sorted(missing)}")
        rows = [r for r in rows if r["id"] in want]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No problems selected after filtering")

    logger.info("Running %d problems", len(rows))

    samples = [
        _make_sample(
            r,
            prefix or "",
            sandbox_memory,
            image=base if staged else None,
            sandbox_disk=sandbox_disk,
            archive=archives[r["id"]],
        )
        for r in rows
    ]
    stage = (
        stage_problem_data(
            {r["id"]: archives[r["id"]] for r in rows},
            data_source,
            s3_uri=s3_uri,
            timeout=stage_timeout,
        )
        if staged and with_sandbox
        else None
    )
    solver = biomystery_solver(
        tool_timeout=tool_timeout,
        message_limit=message_limit,
        token_limit=token_limit,
        max_tool_output=max_tool_output,
        check_data=with_sandbox,
        stage=stage,
    )
    compose = _COMPOSE_DISK if sandbox_disk is not None else _COMPOSE
    return _create_task(
        MemoryDataset(samples),
        solver,
        sandbox=("docker", compose) if with_sandbox else None,
        epochs=epochs,
        metadata={
            "data_source": data_source,
            "base_image": base if staged else None,
            "image_prefix": None if staged else prefix,
            "harness_commit": _harness_commit(),
            "dataset": f"{repo}@{revision}",
        },
    )
