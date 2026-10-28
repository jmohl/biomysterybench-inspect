"""Agentic solver for BioMysteryBench."""

from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    basic_agent,
    solver,
    system_message,
)
from inspect_ai.tool import bash, python, think
from inspect_ai.util import sandbox

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert bioinformatician solving a 'mystery dataset' problem. You have a "
    "sandboxed Linux environment exposed through the `bash` and `python` tools. It is "
    "preloaded with common bioinformatics tools (samtools, bcftools, bedtools, bwa, "
    "bowtie2, minimap2, BLAST+, seqkit, HMMER, MAFFT, fastqc) and Python libraries "
    "(biopython, pysam, pandas, numpy, scipy, scikit-learn). You may install additional "
    "software with `pip` and `conda`/`mamba`, and you may query canonical bioinformatics "
    "databases (NCBI, Ensembl, UCSC, UniProt, EBI) over the network.\n\n"
    "The data files for this problem have already been extracted into the `data/` "
    "subdirectory of your working directory. Inspect and analyse them to answer the "
    "question.\n\n"
    "RULES:\n"
    "- Do NOT look up GEO / SRA / ENA / BioProject accession IDs, the source publication, "
    "or study metadata to identify the original dataset. You must reach the answer by "
    "analysing the data itself. Standard bioinformatics database use (gene/sequence ID "
    "lookup, sequence annotation, reference-genome download, BLAST) IS allowed.\n"
    "- When you are confident, call `submit` with your final answer in EXACTLY the format "
    "the question requests. Put only the answer in `submit` -- not your reasoning."
)

SUBMIT_DESCRIPTION = "Submit your final answer to the question, formatted exactly as the question requests."

# Reply to a submission that was not graded correct, when the agent has
# submissions left. It must never carry the grader's explanation or the rubric.
INCORRECT_SUBMISSION_MESSAGE = (
    "Your submission was incorrect. Please proceed and attempt to find the correct "
    "answer."
)

# Submissions the agent may make per sample. The default, 1, is upstream's
# one-shot protocol.
DEFAULT_MAX_ATTEMPTS = 1

DATA_DIR = "data"

# Name of the agent's submit tool. The scorer reads it to tell whether the
# agent submitted.
SUBMIT_TOOL = "submit"


@solver
def require_problem_data(data_dir: str = DATA_DIR) -> Solver:
    """Raise if the sandbox does not hold the problem's data.

    The task depends on `data/` being populated, by the per-problem image or by
    staging. An empty archive or the wrong image leaves the agent nothing to
    analyse, which is an infrastructure
    failure rather than a model verdict. Raising marks the sample ERRORED, so it is
    eligible for `--retry-on-error` and is counted as a failure, not a wrong answer.

    Args:
        data_dir: directory inside the sandbox that should hold the problem data.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        result = await sandbox().exec(
            ["sh", "-c", f"ls -A {data_dir} 2>/dev/null | head -1"]
        )
        if not result.success or not result.stdout.strip():
            raise RuntimeError(
                f"BioMysteryBench problem {state.sample_id}: '{data_dir}/' is empty or "
                f"missing in image {state.metadata.get('BMB_IMAGE')!r}. The sandbox "
                "did not deliver its data: rebuild the per-problem image with "
                "build_and_push.sh, or check the staging step."
            )
        return state

    return solve


@solver
def biomystery_solver(
    tool_timeout: int = 600,
    message_limit: int | None = 80,
    token_limit: int | None = None,
    max_tool_output: int | None = None,
    system_prompt: str | None = None,
    submit_description: str | None = None,
    check_data: bool = True,
    stage: Solver | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Solver:
    """Open-ended agentic solver for BioMysteryBench.

    The agent may submit up to `max_attempts` times. Every submission but the
    last is graded as soon as it is made, by the task's own scorer. A correct
    one ends the sample. Otherwise the agent is told only that the submission
    was incorrect and keeps working. The last submission is graded at the end
    of the sample like any other.

    Args:
        tool_timeout: per bash/python call timeout in seconds. Alignment, variant
            calling and similar jobs can run for several minutes.
        message_limit: hard cap on messages in the sample. Prevents runaway loops
            on unsolvable samples.
        token_limit: optional token cap for the whole sample.
        max_tool_output: truncate each tool result to this many bytes (guards
            against a `cat` of a huge file blowing the context; None = Inspect
            default).
        system_prompt: override the environment/rules system prompt ("" disables
            it).
        submit_description: override the submit-tool description.
        check_data: verify the sandbox actually contains the problem's data before
            the agent starts. Set False only when running without a sandbox.
        stage: setup solver that puts the problem data into the sandbox before
            the check (see staging.stage_problem_data). None when the image
            already holds the data.
        max_attempts: submissions accepted per sample (1 = one-shot, as
            upstream). The message and token limits are shared by all of them.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be 1 or more, got {max_attempts}")
    resolved_system = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
    init: list[Solver] = []
    if stage is not None:
        init.append(stage)
    if check_data:
        init.append(require_problem_data())
    if resolved_system:
        init.append(system_message(resolved_system))

    return basic_agent(
        init=init,
        tools=[bash(timeout=tool_timeout), python(timeout=tool_timeout), think()],
        max_attempts=max_attempts,
        incorrect_message=INCORRECT_SUBMISSION_MESSAGE,
        message_limit=message_limit,
        token_limit=token_limit,
        max_tool_output=max_tool_output,
        submit_name=SUBMIT_TOOL,
        submit_description=submit_description or SUBMIT_DESCRIPTION,
    )
