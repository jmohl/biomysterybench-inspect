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

Each problem's data is baked into a per-problem sandbox image (built by
build_and_push.sh); a sample's image is named <image_prefix><id> and reaches the
sandbox via Sample.metadata. See README.md for the build and the image layout.
"""

import csv
import logging
import os
import re
from pathlib import Path

from huggingface_hub import hf_hub_download
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.solver import Solver

from biomysterybench.scorer import biomystery_grader
from biomysterybench.solver import biomystery_solver

logger = logging.getLogger(__name__)

FULL_REPO = "Anthropic/BioMysteryBench-full"
PREVIEW_REPO = "Anthropic/BioMysteryBench-preview"
FULL_REVISION = "b5a889c4757214ec9a6ade876b734f920a7799db"
PREVIEW_REVISION = "51c9024021b8989a0cb06ae623b02f90d14c2da3"

_COMPOSE = str(Path(__file__).parent / "compose.yaml")

# A sample's image tag is this prefix + the problem id. Set the prefix (your image
# repo, e.g. registry.example.com/biomystery:) at run time via BMB_IMAGE_PREFIX or
# the image_prefix arg.
_IMAGE_PREFIX_ENV = "BMB_IMAGE_PREFIX"

_DEFAULT_SANDBOX_MEMORY = "32G"
# Docker compose byte values accepted by the k8s_sandbox compose converter.
_MEMORY_RE = re.compile(r"^\d+(\.\d+)?(gb?|mb?|kb?|b)$", re.IGNORECASE)

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
) -> Sample:
    return Sample(
        input=rec["question"],
        target=rec["answer_rubric"],
        id=rec["id"],
        metadata={
            "problem_id": rec["id"],
            "human_solvable": rec.get("human_solvable"),
            "allowed_domains": rec.get("allowed_domains"),
            # Substituted into compose.yaml as ${SAMPLE_METADATA_BMB_IMAGE}.
            "BMB_IMAGE": f"{image_prefix}{rec['id']}",
            # Substituted into compose.yaml as ${SAMPLE_METADATA_BMB_MEMORY}.
            "BMB_MEMORY": sandbox_memory,
        },
    )


def _create_task(
    dataset: Dataset,
    solver: Solver | list[Solver],
    *,
    sandbox: str | tuple[str, str] | None = None,
    epochs: int = 1,
) -> Task:
    """Assemble the Task (shared with tests, which inject a MemoryDataset)."""
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=biomystery_grader(),
        epochs=Epochs(epochs, "mean"),
        sandbox=sandbox,
        version=TASK_VERSION,
    )


@task
def bio_mystery_bench(
    source: str = "full",
    ids: str | list[str] | None = None,
    limit: int | None = None,
    human_solvable_only: bool = False,
    image_prefix: str | None = None,
    sandbox_memory: str = _DEFAULT_SANDBOX_MEMORY,
    with_sandbox: bool = True,
    epochs: int = 1,
    tool_timeout: int = 600,
    message_limit: int | None = 150,
    token_limit: int | None = None,
    max_tool_output: int | None = None,
) -> Task:
    """BioMysteryBench as an Inspect task.

    Each problem's data is baked into a per-problem sandbox image, so the model finds
    it already extracted under `data/`.

    Args:
        source: "full" (90 gated problems) or "preview" (open 5-problem sample).
        ids: restrict to these problem ids. Accepts a list or a single id.
        limit: keep only the first N problems (after filtering).
        human_solvable_only: keep only problems >=1 human solved.
        image_prefix: per-problem image tag prefix; a sample's image is
            image_prefix + problem_id. Defaults to the BMB_IMAGE_PREFIX env var.
        sandbox_memory: memory for each sandbox as a docker compose byte value
            (e.g. "32G", "512M"). On Kubernetes this is both the memory request and
            the memory limit of the sandbox pod.
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

    prefix = (
        image_prefix if image_prefix is not None else os.environ.get(_IMAGE_PREFIX_ENV)
    )
    if with_sandbox and not prefix:
        raise ValueError(
            "image_prefix is unset; pass -T image_prefix=... or set BMB_IMAGE_PREFIX "
            "to your per-problem image repo (see README.md)"
        )

    token = os.environ.get("HF_TOKEN") or None
    repo = FULL_REPO if source == "full" else PREVIEW_REPO
    revision = FULL_REVISION if source == "full" else PREVIEW_REVISION

    rows = _load_problems(repo, revision, token)
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

    samples = [_make_sample(r, prefix or "", sandbox_memory) for r in rows]
    solver = biomystery_solver(
        tool_timeout=tool_timeout,
        message_limit=message_limit,
        token_limit=token_limit,
        max_tool_output=max_tool_output,
        check_data=with_sandbox,
    )
    return _create_task(
        MemoryDataset(samples),
        solver,
        sandbox=("docker", _COMPOSE) if with_sandbox else None,
        epochs=epochs,
    )
