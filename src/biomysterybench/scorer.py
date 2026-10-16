"""Model-graded scorer for BioMysteryBench."""

from inspect_ai.model import ChatMessageTool
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    model_graded_qa,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from biomysterybench.solver import SUBMIT_TOOL

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


@scorer(metrics=[accuracy(), stderr()])
def biomystery_grader(
    template: str | None = None,
    instructions: str | None = None,
    include_history: bool = True,
) -> Scorer:
    """Model-graded BioMysteryBench scorer (rubric plus anti-cheat, transcript-aware).

    Args:
        template: override the grader prompt template.
        instructions: override the grading instructions.
        include_history: give the judge the full transcript (default True; needed
            for the anti-cheat audit). Set False to grade the final answer only
            (cheaper, but cannot detect accession-lookup cheating).

    Score metadata records `submitted`, which is True only when the agent called
    the submit tool. A sample that ran out of messages or tokens first is still
    graded, and the sample's `limit` field in the log records which limit ended
    it. `stop_reason` is the stop reason of the model's last generation, not of
    the sample.

    Note:
        The judge comes from the `grader` model role. Set it with
        `--model-role grader=<model>` or `model_roles`, or grading falls back to
        the target model and the model grades itself.
    """
    inner = model_graded_qa(
        template=template or BMB_GRADER_TEMPLATE,
        instructions=instructions or BMB_GRADER_INSTRUCTIONS,
        include_history=include_history,
        partial_credit=False,
        model_role="grader",
    )

    async def score(state: TaskState, target: Target) -> Score | None:
        result = await inner(state, target)
        if result is None:
            return None
        meta = dict(result.metadata or {})
        meta["human_solvable"] = state.metadata.get("human_solvable")
        meta["problem_id"] = state.metadata.get("problem_id") or state.sample_id
        meta["submitted"] = submitted(state)
        meta["stop_reason"] = state.output.stop_reason if state.output else None
        # Copy so every other field of the inner score survives, including why
        # a sample the grader could not judge was left unscored.
        return result.model_copy(update={"metadata": meta})

    return score


def submitted(state: TaskState) -> bool:
    """Whether the agent called the submit tool during the sample."""
    return any(
        isinstance(m, ChatMessageTool) and m.function == SUBMIT_TOOL
        for m in state.messages
    )
