"""Model-graded scorer for BioMysteryBench."""

import math
from copy import copy

from inspect_ai.model import ChatMessageTool
from inspect_ai.scorer import (
    CORRECT,
    Score,
    Scorer,
    Target,
    Value,
    accuracy,
    model_graded_qa,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox, suspend_token_limit

from biomysterybench.solver import DEFAULT_MAX_ATTEMPTS, SUBMIT_TOOL

# The grader is called again when it returns no grade (no parseable GRADE line,
# or a content_filter block, which can clear on a retry), up to this many calls.
MAX_GRADER_ATTEMPTS = 3

# Sample metadata key listing the grade of each submission graded mid-run, in
# order. An unscored grade is recorded as None.
ATTEMPTS_KEY = "bmb_attempts"

# Reads the sandbox's own cgroup (v2, then v1) and the size of /workspace.
# memory.peak includes page cache, so it is an upper bound on what the agent's
# processes needed; oom_kill counts processes the kernel killed at the limit.
_TELEMETRY_SCRIPT = r"""
cg=/sys/fs/cgroup
peak=$(cat $cg/memory.peak 2>/dev/null || cat $cg/memory/memory.max_usage_in_bytes 2>/dev/null || echo -)
oom=$(awk '$1 == "oom_kill" {print $2}' $cg/memory.events 2>/dev/null)
ws=$(du -sb /workspace 2>/dev/null | cut -f1)
echo "${peak:--} ${oom:--} ${ws:--}"
"""

# include_history=True puts the transcript in {question}, not {answer}.
BMB_GRADER_TEMPLATE = """
You are grading a bioinformatician's attempt at an open-ended "mystery dataset" problem.
The model was given raw biological data files and had to analyse them to answer.

[BEGIN DATA]
***
[Question and full transcript of the model's work]:
{question}
***
[Final submitted answer]: {answer}
***
[Answer key / grading rubric]: {criterion}
***
[END DATA]

{instructions}
"""

BMB_GRADER_INSTRUCTIONS = (
    "Grade the attempt in two parts.\n"
    "1. CORRECTNESS: Does the final submitted answer match the expected answer in the "
    "rubric? Accept scientifically equivalent phrasings, synonyms, and correct "
    "nomenclature; require the specific value/name the rubric asks for.\n"
    "2. NO CHEATING: From the transcript, did the model reach the answer by ANALYSING the "
    "data? It is CHEATING (and disqualifies the answer) if the model identified the source "
    "dataset by looking up a GEO / SRA / ENA / BioProject accession ID, the source "
    "publication, or study metadata. Standard database use -- gene/sequence ID lookup, "
    "sequence annotation, reference-genome download, BLAST -- is NOT cheating.\n\n"
    "First give one or two sentences of reasoning covering both parts. Then, on a new line, "
    "write exactly 'GRADE: C' if the answer is correct AND the model did not cheat, or "
    "'GRADE: I' otherwise."
)

# Prepended to the instructions when the agent may submit more than once.
BMB_MULTI_ATTEMPT_NOTE = (
    "The model was allowed several submissions. After each submission but the last "
    "it was told only that the submission was incorrect, so any earlier submission in "
    "the transcript was judged incorrect. Grade only the final submitted answer "
    "above, and apply the cheating check to the whole transcript, including the work "
    "before earlier submissions.\n\n"
)


@scorer(metrics=[accuracy(), stderr()])
def biomystery_grader(
    template: str | None = None,
    instructions: str | None = None,
    include_history: bool = True,
    telemetry: bool = True,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Scorer:
    """Model-graded BioMysteryBench scorer (rubric plus anti-cheat, transcript-aware).

    Args:
        template: override the grader prompt template.
        instructions: override the grading instructions.
        include_history: give the judge the full transcript (default True; needed
            for the anti-cheat audit). Set False to grade the final answer only
            (cheaper, but cannot detect accession-lookup cheating).
        telemetry: record the sandbox's peak memory, OOM kills and /workspace
            size in the score metadata, for sizing sandboxes. Needs a sandbox.
        max_attempts: submissions the solver accepts. Above 1, the grader is
            told that earlier submissions in the transcript were judged
            incorrect, and the default instructions say so.

    The grader is called again, up to MAX_GRADER_ATTEMPTS calls, while it
    returns no grade. Its tokens do not count toward the agent's token limit.
    The solver calls this scorer mid-run for every submission but the last, and
    each of those grades is recorded in the sample metadata under ATTEMPTS_KEY.

    Score metadata records:
        submitted: True when the agent called the submit tool. A sample that
            ran out of messages or tokens first is still graded, and the
            sample's `limit` field in the log records which limit ended it.
        n_submissions: the number of submit calls.
        attempt_grades: the grades given mid-run, in order, then this grade.
            An unscored grade is None. A submission graded correct mid-run is
            graded again at the end, so it appears twice.
        first_attempt_correct: whether the first graded answer was correct,
            for comparison with one-shot runs. None when it was unscored.
        graded_last_submission: True when the sample ended (on a limit or a
            context overflow) after a submission without a new one, so the
            last submission was graded instead of the model's last text.
        grader_attempts: grader calls used for this grade.
        stop_reason: the stop reason of the model's last generation, not of
            the sample.

    Note:
        The judge comes from the `grader` model role. Set it with
        `--model-role grader=<model>` or `model_roles`, or grading falls back to
        the target model and the model grades itself.
    """
    if instructions is None:
        instructions = BMB_GRADER_INSTRUCTIONS
        if max_attempts > 1:
            instructions = BMB_MULTI_ATTEMPT_NOTE + instructions
    inner = model_graded_qa(
        template=template or BMB_GRADER_TEMPLATE,
        instructions=instructions,
        include_history=include_history,
        partial_credit=False,
        model_role="grader",
    )

    async def score(state: TaskState, target: Target) -> Score | None:
        answers = submissions(state)
        # After a limit or a context overflow, output.completion is the model's
        # last text, not an answer. Grade the last submission instead, on a copy
        # so that the logged output is unchanged.
        graded_last = bool(answers) and answers[-1] != state.output.completion
        graded = _with_completion(state, answers[-1]) if graded_last else state

        # Mid-run, grader tokens would otherwise count toward the agent's limit.
        with suspend_token_limit():
            for grader_attempts in range(1, MAX_GRADER_ATTEMPTS + 1):
                result = await inner(graded, target)
                if result is None or _grade(result.value) is not None:
                    break
        if result is None:
            return None

        grade = _grade(result.value)
        earlier = list(state.metadata.get(ATTEMPTS_KEY, []))
        # Inspect marks the state completed before the end-of-sample score, so
        # an uncompleted state means the solver is grading a submission mid-run.
        if not state.completed:
            state.metadata[ATTEMPTS_KEY] = earlier + [grade]
        attempt_grades = earlier + [grade]

        meta = dict(result.metadata or {})
        meta["human_solvable"] = state.metadata.get("human_solvable")
        meta["problem_id"] = state.metadata.get("problem_id") or state.sample_id
        meta["submitted"] = bool(answers)
        meta["n_submissions"] = len(answers)
        meta["attempt_grades"] = attempt_grades
        meta["first_attempt_correct"] = (
            None if attempt_grades[0] is None else attempt_grades[0] == CORRECT
        )
        meta["graded_last_submission"] = graded_last
        meta["grader_attempts"] = grader_attempts
        meta["stop_reason"] = state.output.stop_reason if state.output else None
        if telemetry:
            meta["sandbox_resources"] = await sandbox_resources()
        # Copy so every other field of the inner score survives, including why
        # a sample the grader could not judge was left unscored.
        return result.model_copy(update={"metadata": meta})

    return score


def submissions(state: TaskState) -> list[str]:
    """The text of each submit tool call in the sample, in order."""
    return [
        m.text
        for m in state.messages
        if isinstance(m, ChatMessageTool) and m.function == SUBMIT_TOOL
    ]


def _with_completion(state: TaskState, answer: str) -> TaskState:
    """A shallow copy of `state` whose output completion is `answer`."""
    graded = copy(state)
    graded.output = state.output.model_copy(update={"completion": answer})
    return graded


def _grade(value: Value) -> Value | None:
    """A score value, or None when the score is unscored (NaN)."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


async def sandbox_resources() -> dict[str, int | None] | None:
    """Peak memory, OOM kills and /workspace size of the sample's sandbox.

    Read at scoring time, so it covers the whole sample, including samples that
    ended on a message or token limit. Returns None if the sandbox cannot be
    read; a value the sandbox does not expose is None.
    """
    try:
        result = await sandbox().exec(["sh", "-c", _TELEMETRY_SCRIPT], timeout=600)
    except Exception:  # noqa: BLE001
        return None
    fields = result.stdout.split() if result.success else []
    if len(fields) != 3:
        return None
    peak, oom, workspace = (int(f) if f.isdigit() else None for f in fields)
    return {"memory_peak_bytes": peak, "oom_kills": oom, "workspace_bytes": workspace}
