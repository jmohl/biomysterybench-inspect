"""Offline tests for BioMysteryBench (no HuggingFace / no model calls).

Covers the solver's construction, the model-graded scorer's metadata stamping (grader
model mocked), sample staging logic, and task wiring (agentic solver + grader + sandbox)
via an in-memory dataset. End-to-end behaviour (gated dataset download, real model,
sandbox execution, real judging) is exercised by a gated smoke run, not here.
"""

import asyncio
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from inspect_ai import eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalLog
from inspect_ai.model import ModelOutput, ModelUsage, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, Score
from inspect_ai.solver import Generate, TaskState, solver

import biomysterybench.biomysterybench as bmb
import biomysterybench.scorer as scoring
import biomysterybench.solver as solving
from biomysterybench import bio_mystery_bench
from biomysterybench.solver import biomystery_solver, require_problem_data


@solver
def mock_solver_with_output(output: str):
    """Set a fixed model output without calling a model."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output = ModelOutput.from_content(model="mockllm/model", content=output)
        return state

    return solve


def assert_eval_success(log: EvalLog, min_samples: int) -> None:
    assert log.status == "success", f"eval failed: {log.error}"
    assert log.samples is not None and len(log.samples) >= min_samples
    assert log.results is not None and log.results.scores


# --- solver / scorer / task construct ----------------------------------------


def test_solver_builds():
    assert callable(biomystery_solver(tool_timeout=30, message_limit=5))


def _sandbox_returning(stdout: str, success: bool = True):
    """Stub inspect_ai's sandbox() with one whose exec returns a fixed result."""

    async def fake_exec(cmd):
        return SimpleNamespace(success=success, stdout=stdout, stderr="")

    return lambda *a, **k: SimpleNamespace(exec=fake_exec)


def test_require_problem_data_passes_when_data_present(monkeypatch):
    monkeypatch.setattr(solving, "sandbox", _sandbox_returning("reads.fastq\n"))
    state = SimpleNamespace(sample_id="hb002", metadata={"BMB_IMAGE": "repo:hb002"})
    result = asyncio.run(require_problem_data()(state, None))
    assert result is state


def test_require_problem_data_raises_when_data_missing(monkeypatch):
    """An empty data/ is an infrastructure failure, not a wrong answer.

    A bad image must surface as an ERRORED sample rather than as a low score.
    """
    monkeypatch.setattr(solving, "sandbox", _sandbox_returning(""))
    state = SimpleNamespace(sample_id="hb002", metadata={"BMB_IMAGE": "repo:hb002"})
    with pytest.raises(RuntimeError, match="did not deliver its data"):
        asyncio.run(require_problem_data()(state, None))


def test_require_problem_data_raises_when_ls_fails(monkeypatch):
    monkeypatch.setattr(solving, "sandbox", _sandbox_returning("x", success=False))
    state = SimpleNamespace(sample_id="hb002", metadata={"BMB_IMAGE": "repo:hb002"})
    with pytest.raises(RuntimeError, match="did not deliver its data"):
        asyncio.run(require_problem_data()(state, None))


def test_task_is_callable_and_registered():
    # @task-decorated; importable and carries a registry name
    assert callable(bio_mystery_bench)


# --- scorer metadata stamping (grader model mocked) --------------------------


def _fake_state(human_solvable="yes", pid="hb002"):
    return SimpleNamespace(
        output=SimpleNamespace(completion="Bacillus licheniformis", stop_reason="stop"),
        messages=[],
        metadata={"human_solvable": human_solvable, "problem_id": pid},
        sample_id=pid,
        completed=True,
    )


def test_grader_stamps_metadata(monkeypatch):
    def fake_model_graded_qa(**kwargs):
        async def inner(state, target):
            return Score(
                value=CORRECT,
                answer=state.output.completion,
                explanation="matches rubric",
                metadata={"grader_model": "fake"},
            )

        return inner

    monkeypatch.setattr(scoring, "model_graded_qa", fake_model_graded_qa)
    result = asyncio.run(
        scoring.biomystery_grader()(_fake_state(), SimpleNamespace(text="rubric"))
    )
    assert result.value == CORRECT
    assert result.metadata["human_solvable"] == "yes"
    assert result.metadata["problem_id"] == "hb002"
    assert result.metadata["stop_reason"] == "stop"
    assert result.metadata["submitted"] is False
    assert result.metadata["grader_model"] == "fake"  # inner metadata preserved


def test_unsubmitted_sample_is_flagged_and_its_limit_logged(tmp_path):
    """A sample that hits message_limit without submitting is still graded.

    Its score records submitted=False, and the log's sample limit records the
    message limit. The model's stop_reason cannot show this, because it
    describes only the last generation.
    """

    def agent(messages, tools, tool_choice, config):
        if any("Answer now" in m.text for m in messages if m.role == "user"):
            return ModelOutput.for_tool_call(
                "mockllm/model", "submit", {"answer": "Bacillus licheniformis"}
            )
        return ModelOutput.from_content("mockllm/model", "Still analysing the data.")

    def grader(messages, tools, tool_choice, config):
        return ModelOutput.from_content("mockllm/model", "Reasoning.\nGRADE: I")

    dataset = MemoryDataset(
        [
            Sample(input="Answer now.", target="rubric", id="answers"),
            Sample(input="Keep working.", target="rubric", id="stalls"),
        ]
    )
    task = bmb._create_task(
        dataset,
        biomystery_solver(message_limit=6, check_data=False, max_attempts=1),
        max_attempts=1,
    )

    [log] = eval(
        tasks=[task],
        model=get_model("mockllm/model", custom_outputs=agent),
        model_roles={"grader": get_model("mockllm/model", custom_outputs=grader)},
        log_dir=str(tmp_path),
    )

    assert_eval_success(log, min_samples=2)
    assert log.samples is not None
    samples = {s.id: s for s in log.samples}
    answered, stalled = samples["answers"], samples["stalls"]
    assert answered.scores is not None and stalled.scores is not None
    assert answered.scores["biomystery_grader"].metadata["submitted"] is True
    assert answered.limit is None
    assert stalled.scores["biomystery_grader"].metadata["submitted"] is False
    assert stalled.scores["biomystery_grader"].value == INCORRECT
    assert stalled.limit is not None and stalled.limit.type == "message"


# --- multiple submissions (mock agent and grader) ----------------------------

RIGHT, WRONG = "Bacillus licheniformis", "Bacillus subtilis"


def _submitting_agent(answers):
    """Agent that submits answers[i] as its i-th submission, then keeps talking."""

    def agent(messages, tools, tool_choice, config):
        n = sum(1 for m in messages if getattr(m, "function", None) == "submit")
        if n < len(answers):
            return ModelOutput.for_tool_call(
                "mockllm/model", "submit", {"answer": answers[n]}
            )
        return ModelOutput.from_content("mockllm/model", "Still analysing the data.")

    return agent


def _grader_accepting(answer):
    """Grader that grades C exactly when the submitted answer is `answer`."""

    def grader(messages, tools, tool_choice, config):
        ok = f"[Final submitted answer]: {answer}\n" in messages[-1].text
        return ModelOutput.from_content(
            "mockllm/model", f"Reasoning.\nGRADE: {'C' if ok else 'I'}"
        )

    return grader


def _run_attempts(tmp_path, agent, grader, max_attempts, epochs=1, **solver_args):
    dataset = MemoryDataset([Sample(input="Which organism?", target="rubric", id="x")])
    solver_args.setdefault("message_limit", 40)
    task = bmb._create_task(
        dataset,
        biomystery_solver(check_data=False, max_attempts=max_attempts, **solver_args),
        epochs=epochs,
        max_attempts=max_attempts,
    )
    [log] = eval(
        tasks=[task],
        model=get_model("mockllm/model", custom_outputs=agent),
        model_roles={"grader": get_model("mockllm/model", custom_outputs=grader)},
        log_dir=str(tmp_path),
    )
    assert_eval_success(log, min_samples=epochs)
    assert log.samples is not None
    return log.samples


def _intermediate_scores(sample):
    return [e for e in sample.events if e.event == "score" and e.intermediate]


def _incorrect_replies(sample):
    return [
        m
        for m in sample.messages
        if m.role == "user" and m.text == solving.INCORRECT_SUBMISSION_MESSAGE
    ]


def test_wrong_then_right_submission_stops_the_loop(tmp_path):
    [sample] = _run_attempts(
        tmp_path, _submitting_agent([WRONG, RIGHT]), _grader_accepting(RIGHT), 5
    )
    score = sample.scores["biomystery_grader"]
    assert score.value == CORRECT
    assert score.metadata["n_submissions"] == 2
    # Both submissions graded mid-run, then the correct one graded again.
    assert score.metadata["attempt_grades"] == [INCORRECT, CORRECT, CORRECT]
    assert score.metadata["first_attempt_correct"] is False
    assert score.metadata["graded_last_submission"] is False
    assert len(_intermediate_scores(sample)) == 2
    # The loop ends on the correct submission.
    assert sample.messages[-1].role == "tool"
    # The model hears only the generic message, never the grader's reasoning.
    [reply] = _incorrect_replies(sample)
    assert "Reasoning" not in reply.text and "GRADE" not in reply.text


def test_always_wrong_stops_after_max_attempts(tmp_path):
    [sample] = _run_attempts(
        tmp_path, _submitting_agent([WRONG] * 10), _grader_accepting(RIGHT), 5
    )
    score = sample.scores["biomystery_grader"]
    assert score.value == INCORRECT
    assert score.metadata["n_submissions"] == 5
    # The 5th submission is graded only at the end of the sample.
    assert len(_intermediate_scores(sample)) == 4
    assert len(_incorrect_replies(sample)) == 4
    assert score.metadata["attempt_grades"] == [INCORRECT] * 5
    assert sample.limit is None


def test_one_attempt_keeps_the_one_shot_protocol(tmp_path):
    [sample] = _run_attempts(
        tmp_path, _submitting_agent([WRONG, RIGHT]), _grader_accepting(RIGHT), 1
    )
    score = sample.scores["biomystery_grader"]
    assert score.value == INCORRECT
    assert score.metadata["n_submissions"] == 1
    assert score.metadata["attempt_grades"] == [INCORRECT]
    assert score.metadata["first_attempt_correct"] is False
    assert _intermediate_scores(sample) == []
    assert _incorrect_replies(sample) == []
    assert scoring.ATTEMPTS_KEY not in sample.metadata


@pytest.mark.parametrize("max_attempts", [1, 2])
def test_epochs_are_independent_runs(tmp_path, max_attempts):
    """max_attempts=1 with epochs=k gives k independent one-shot runs (pass@k).

    Each epoch keeps its own attempt record; nothing carries over between them.
    """
    samples = _run_attempts(
        tmp_path,
        _submitting_agent([WRONG] * 10),
        _grader_accepting(RIGHT),
        max_attempts,
        epochs=2,
    )
    assert sorted(s.epoch for s in samples) == [1, 2]
    for sample in samples:
        score = sample.scores["biomystery_grader"]
        assert score.metadata["n_submissions"] == max_attempts
        assert score.metadata["attempt_grades"] == [INCORRECT] * max_attempts
        assert len(_intermediate_scores(sample)) == max_attempts - 1


def test_limit_after_a_wrong_submission_grades_the_last_submission(tmp_path):
    """The message limit ends the sample after a wrong submission.

    The model's last text is not an answer, so the grader is given the last
    submission. The logged output and transcript are left as they were.
    """
    [sample] = _run_attempts(
        tmp_path,
        _submitting_agent([WRONG]),
        _grader_accepting(RIGHT),
        5,
        message_limit=8,
    )
    assert sample.limit is not None and sample.limit.type == "message"
    score = sample.scores["biomystery_grader"]
    # Score.answer is the answer the grader was given.
    assert score.answer == WRONG
    assert score.value == INCORRECT
    assert score.metadata["graded_last_submission"] is True
    assert score.metadata["n_submissions"] == 1
    assert score.metadata["attempt_grades"] == [INCORRECT, INCORRECT]
    assert sample.output.completion == "Still analysing the data."
    assert sample.messages[-1].role == "assistant"
    assert sample.messages[-1].text == "Still analysing the data."


def _grader_outputs(*completions):
    return [ModelOutput.from_content("mockllm/model", c) for c in completions]


def test_grader_is_asked_again_when_it_gives_no_grade(tmp_path):
    [sample] = _run_attempts(
        tmp_path,
        _submitting_agent([RIGHT]),
        _grader_outputs("No verdict.", "No verdict.", "Reasoning.\nGRADE: C"),
        1,
    )
    score = sample.scores["biomystery_grader"]
    assert score.value == CORRECT
    assert score.metadata["grader_attempts"] == 3


def test_unscored_intermediate_grade_counts_as_incorrect(tmp_path):
    """A mid-run grade the grader never gave tells the agent it was wrong."""
    [sample] = _run_attempts(
        tmp_path,
        _submitting_agent([RIGHT, RIGHT]),
        _grader_outputs(
            *["No verdict."] * scoring.MAX_GRADER_ATTEMPTS,
            "Reasoning.\nGRADE: C",
            "Reasoning.\nGRADE: C",
        ),
        5,
    )
    score = sample.scores["biomystery_grader"]
    assert score.value == CORRECT
    assert score.metadata["attempt_grades"] == [None, CORRECT, CORRECT]
    assert score.metadata["first_attempt_correct"] is None
    assert len(_incorrect_replies(sample)) == 1


def test_grader_tokens_do_not_count_toward_the_token_limit(tmp_path):
    """Mid-run grades use far more tokens than the limit and do not end the run."""

    def grader(messages, tools, tool_choice, config):
        output = ModelOutput.from_content("mockllm/model", "Reasoning.\nGRADE: I")
        output.usage = ModelUsage(
            input_tokens=1_000_000, output_tokens=10, total_tokens=1_000_010
        )
        return output

    [sample] = _run_attempts(
        tmp_path,
        _submitting_agent([WRONG] * 10),
        grader,
        3,
        token_limit=100_000,
    )
    assert sample.limit is None
    assert sample.scores["biomystery_grader"].metadata["n_submissions"] == 3


def test_task_defaults_to_one_shot(monkeypatch):
    """Without max_attempts the task keeps upstream's one-shot protocol."""
    monkeypatch.setattr(bmb, "_load_problems", lambda *a, **k: [_REC])
    task = bio_mystery_bench(source="preview", image_prefix="repo:")
    assert task.metadata["max_attempts"] == 1


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError, match="max_attempts must be 1 or more"):
        biomystery_solver(max_attempts=0)
    with pytest.raises(ValueError, match="max_attempts must be 1 or more"):
        bio_mystery_bench(source="preview", max_attempts=0, with_sandbox=False)


@pytest.mark.parametrize("max_attempts, noted", [(1, False), (5, True)])
def test_grader_is_told_about_earlier_submissions(monkeypatch, max_attempts, noted):
    captured = {}

    def fake_model_graded_qa(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(scoring, "model_graded_qa", fake_model_graded_qa)
    scoring.biomystery_grader(max_attempts=max_attempts)
    assert (scoring.BMB_MULTI_ATTEMPT_NOTE in captured["instructions"]) is noted
    assert captured["instructions"].endswith(scoring.BMB_GRADER_INSTRUCTIONS)


# --- sample mapping + selection (no network) ---------------------------------

# Adapted from problem hb002 of Anthropic/BioMysteryBench-preview, (c) 2026
# Anthropic, PBC, licensed CC BY 4.0
# (https://creativecommons.org/licenses/by/4.0/). The question, rubric and
# allowed_domains are shortened here.
_REC = {
    "id": "hb002",
    "question": "What bacteria is found in this sequenced dataset?",
    "answer_rubric": "The answer is Bacillus licheniformis. Score 1.0 if correct.",
    "allowed_domains": "ncbi.nlm.nih.gov, ensembl.org",
    "human_solvable": "yes",
}


def test_make_sample_maps_fields():
    s = bmb._make_sample(_REC, "registry.example.com/biomystery:")
    assert isinstance(s, Sample)
    assert s.input == _REC["question"]
    assert s.target == _REC["answer_rubric"]  # rubric is the grader's criterion
    assert s.id == "hb002"
    assert s.metadata["human_solvable"] == "yes"
    assert s.metadata["allowed_domains"] == "ncbi.nlm.nih.gov, ensembl.org"
    # image tag = prefix + problem id; substituted into compose as SAMPLE_METADATA_BMB_IMAGE
    assert s.metadata["BMB_IMAGE"] == "registry.example.com/biomystery:hb002"
    # no eager data staging: images already carry the data
    assert s.files is None and s.setup is None


def test_make_sample_carries_sandbox_memory():
    s = bmb._make_sample(_REC, "repo:", sandbox_memory="48G")
    assert s.metadata["BMB_MEMORY"] == "48G"


def test_compose_sets_memory_limit_from_sample_metadata():
    """The runner substitutes BMB_MEMORY into mem_limit.

    The k8s compose converter reads mem_limit and sets both the memory request and
    the memory limit of the pod to this value.
    """
    s = bmb._make_sample(_REC, "repo:", sandbox_memory="48G")
    text = Path(bmb._COMPOSE).read_text()
    for key, value in s.metadata.items():
        text = text.replace(f"${{SAMPLE_METADATA_{key.upper()}}}", str(value))
    service = yaml.safe_load(text)["services"]["default"]
    assert service["mem_limit"] == "48G"
    assert service["image"] == "repo:hb002"


@pytest.mark.parametrize("compose", [bmb._COMPOSE, bmb._COMPOSE_DISK])
def test_compose_files_are_named_for_k8s_conversion(compose):
    """k8s_sandbox converts a file only if its name ends in compose.yaml or compose.yml.

    Any other name is passed to Helm as raw values, which the chart rejects.
    """
    assert Path(compose).name.endswith(("compose.yaml", "compose.yml"))


def test_compose_sets_fixed_cpu_limit():
    """The k8s compose converter turns cpus into the pod's CPU request and limit."""
    service = yaml.safe_load(Path(bmb._COMPOSE).read_text())["services"]["default"]
    assert service["cpus"] == 4


def test_task_defaults_sandbox_memory_to_32g(monkeypatch):
    monkeypatch.setattr(bmb, "_load_problems", lambda *a, **k: [_REC])
    task = bio_mystery_bench(source="preview", image_prefix="repo:")
    assert [s.metadata["BMB_MEMORY"] for s in task.dataset] == ["32G"]


@pytest.mark.parametrize("value", ["16Gi", "16", "lots", ""])
def test_task_rejects_bad_sandbox_memory(value):
    with pytest.raises(ValueError, match="sandbox_memory"):
        bio_mystery_bench(source="preview", image_prefix="repo:", sandbox_memory=value)


# --- task wiring (no HF: inject a MemoryDataset) -----------------------------


def test_create_task_declares_sandbox_and_grader():
    task = bmb._create_task(
        MemoryDataset([Sample(input="q", target="rubric", id="x")]),
        biomystery_solver(),
        sandbox=("docker", "compose.yaml"),
    )
    assert task.sandbox is not None
    assert getattr(task.sandbox, "type", None) == "docker"
    assert task.scorer is not None


def test_task_rejects_negative_limit():
    with pytest.raises(ValueError, match="limit must be zero or more"):
        bio_mystery_bench(source="preview", limit=-1, with_sandbox=False)


def test_task_rejects_bad_source():
    with pytest.raises(ValueError, match="source must be"):
        bio_mystery_bench(source="nope", with_sandbox=False)


def test_ids_accepts_a_bare_string(monkeypatch):
    """`-T ids=hb002` reaches the task as a string, not a one-element list.

    Inspect's CLI collapses a single-element list back to a bare string. Treating
    that as a sequence gives a set of characters, so nothing matches and the task
    raises "No problems selected after filtering" for a valid id.
    """
    rows = [
        {"id": "hb002", "question": "q", "answer_rubric": "r", "human_solvable": "yes"},
        {"id": "hb020", "question": "q", "answer_rubric": "r", "human_solvable": "yes"},
    ]
    monkeypatch.setattr(bmb, "_load_problems", lambda *a, **k: rows)
    task = bio_mystery_bench(
        source="preview", ids="hb002", with_sandbox=False, image_prefix="repo:"
    )
    assert [s.id for s in task.dataset] == ["hb002"]


def test_ids_reports_unknown_ids(monkeypatch):
    rows = [
        {"id": "hb002", "question": "q", "answer_rubric": "r", "human_solvable": "yes"}
    ]
    monkeypatch.setattr(bmb, "_load_problems", lambda *a, **k: rows)
    with pytest.raises(ValueError, match="ids not in the dataset"):
        bio_mystery_bench(
            source="preview",
            ids=["hb002", "hb0O2"],
            with_sandbox=False,
            image_prefix="repo:",
        )


def test_task_requires_image_prefix_when_sandboxed(monkeypatch):
    monkeypatch.delenv("BMB_IMAGE_PREFIX", raising=False)
    with pytest.raises(ValueError, match="image_prefix is unset"):
        bio_mystery_bench(source="preview", with_sandbox=True, image_prefix=None)


def test_end_to_end_with_mockllm(tmp_path):
    """Run one sample end to end with a mock model, no sandbox and no network."""
    dataset = MemoryDataset(
        [
            Sample(
                input="Which organism does this genome belong to?",
                target=(
                    "The answer is Bacillus licheniformis. Score 1.0 if the model "
                    "did not cheat AND got the answer correct. Score 0 otherwise."
                ),
                id="hb002",
                metadata={"problem_id": "hb002", "human_solvable": "yes"},
            )
        ]
    )
    task = bmb._create_task(dataset, mock_solver_with_output("Bacillus licheniformis"))

    [log] = eval(tasks=[task], model="mockllm/model", limit=1, log_dir=str(tmp_path))

    assert_eval_success(log, min_samples=1)
    assert log.samples is not None
    assert log.samples[0].scores is not None


def test_grader_refusal_leaves_sample_unscored(tmp_path):
    """A grader that returns no verdict leaves the sample out of accuracy.

    The grader's safety classifier can stop it on biological content with
    stop_reason content_filter and an empty completion. That sample was never
    judged, so it must be unscored (NaN, excluded from metrics), not INCORRECT.
    This relies on inspect-ai 0.3.245 or later. The grader is asked
    MAX_GRADER_ATTEMPTS times before the sample is left unscored.
    """
    dataset = MemoryDataset(
        [
            Sample(input="q", target="rubric", id=pid, metadata={"problem_id": pid})
            for pid in ("hb002", "hb019")
        ]
    )
    task = bmb._create_task(dataset, mock_solver_with_output("Ebola virus"))
    refusal = ModelOutput.from_content(
        model="mockllm/model", content="", stop_reason="content_filter"
    )
    grader = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content(model="mockllm/model", content="GRADE: C"),
            *[refusal] * scoring.MAX_GRADER_ATTEMPTS,
        ],
    )

    [log] = eval(
        tasks=[task],
        model="mockllm/model",
        model_roles={"grader": grader},
        max_samples=1,
        log_dir=str(tmp_path),
    )

    assert_eval_success(log, min_samples=2)
    assert log.samples is not None
    scores = {s.id: s.scores["biomystery_grader"] for s in log.samples}
    assert scores["hb002"].value == CORRECT
    refused = scores["hb019"]
    assert math.isnan(refused.value)
    assert refused.explanation.startswith("Grade not found")
    # Newer inspect-ai releases record why the sample is unscored in Score.reason,
    # older ones in metadata["unscored_reason"].
    assert getattr(refused, "reason", None) or refused.metadata.get("unscored_reason")
    assert refused.metadata["problem_id"] == "hb019"
    assert refused.metadata["grader_attempts"] == scoring.MAX_GRADER_ATTEMPTS
    assert log.results is not None
    [result] = log.results.scores
    assert result.scored_samples == 1
    assert result.unscored_samples == 1
    assert result.metrics["accuracy"].value == 1.0
