# BioMysteryBench for Inspect

An [Inspect AI](https://inspect.aisi.org.uk/) implementation of [BioMysteryBench](https://www.anthropic.com/research/Evaluating-Claude-For-Bioinformatics-With-BioMysteryBench), Anthropic's open-ended agentic bioinformatics benchmark.

Each problem gives the model a real, anonymised biological dataset (DNA and RNA sequencing, proteomics, metabolomics, methylation, ChIP-seq and others) and asks a question that can only be answered by analysing the data. The model works in a Linux sandbox with common bioinformatics tools and internet access to NCBI, Ensembl, UniProt and the conda and pip channels. It submits a free-form answer, and a model judge grades that answer against the problem's rubric after reading the full transcript.

The benchmark, its problems and its rubrics are Anthropic's work. This repository contains only the Inspect implementation.

## Requirements

- Python 3.10 or later and [uv](https://docs.astral.sh/uv/).
- `inspect_ai` 0.3.245 or later. The package requires it, and the [Scoring](#scoring) section explains why.
- Docker, or a Kubernetes cluster with [inspect_k8s_sandbox](https://k8s-sandbox.aisi.org.uk/). Each sandbox uses 4 CPUs and 32 GB of memory by default.
- A container registry you can push to and pull from. The per-problem images total about 155 GB for the full set. With [run-time staging](#staging-data-at-run-time) only the base image (about 5 GB, no benchmark data) goes in a registry.
- A HuggingFace token with access to the gated full dataset (see [Dataset access and terms](#dataset-access-and-terms)). The preview set needs no token, either to build its images or to run it.
- API access to a grader model and to the model under test, and the provider's Python SDK (see [Installation](#installation)).

## Installation

```bash
git clone https://github.com/safer-ai/biomysterybench-inspect
cd biomysterybench-inspect
uv sync --extra openai --extra anthropic
```

Inspect calls each model provider through that provider's own SDK, which the package does not install by default. The `openai` and `anthropic` extras install the SDKs the examples below use. For another provider, add its SDK with `uv pip install <package>` (the [Inspect providers page](https://inspect.aisi.org.uk/providers.html) lists them). To run the sandboxes on Kubernetes, also add the `k8s` extra, which installs [inspect_k8s_sandbox](https://k8s-sandbox.aisi.org.uk/):

```bash
uv sync --extra openai --extra anthropic --extra k8s
```

The package is named `biomysterybench` and registers the task with Inspect as `biomysterybench/bio_mystery_bench`.

## Dataset access and terms

The task reads two HuggingFace datasets, each pinned to a fixed revision in `src/biomysterybench/biomysterybench.py`:

- [`Anthropic/BioMysteryBench-full`](https://huggingface.co/datasets/Anthropic/BioMysteryBench-full) has 90 problems, one `data/<id>.zip` per problem. It is 155 GB in total. The median archive is about 56 MB, but six exceed 10 GB and the largest is about 27 GB. The dataset is gated. Request access on its HuggingFace page, then set `HF_TOKEN` when building images and when running the task.
- [`Anthropic/BioMysteryBench-preview`](https://huggingface.co/datasets/Anthropic/BioMysteryBench-preview) has 5 open problems in a single 11 MB `data.zip`. It is not gated and is the quickest way to check a setup end to end.

Three sets of terms apply, and they cover different things:

- **This repository's code** is MIT-licensed (see [Licence and attribution](#licence-and-attribution)).
- **The benchmark text**, meaning the problem statements, answer rubrics and task formulation, is © Anthropic and licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (see the dataset's `LICENSE` file). The full dataset's access agreement adds conditions. It allows the benchmark to be used for evaluation and benchmarking only. It forbids using the problem statements, answer rubrics or task formulation to train, fine-tune, reinforce or distil any model. It permits building evaluation prompts at inference time and publishing results with attribution to Anthropic. Read the agreement on the dataset page before requesting access.
- **The biological data** in `data.zip` and `data/<id>.zip` are anonymised derivatives of data first deposited in public archives such as NCBI GEO, SRA, ENA and ArrayExpress. According to the dataset's `LICENSE` file, they remain subject to the data-use policies of those original repositories, and Anthropic claims no additional rights over them. The per-problem images contain this data. Check those policies before pushing the images to a public registry or sharing them.

Eval logs contain the problem statements and rubrics. Treat logs from the full set as covered by the access agreement before sharing them.

The only benchmark content in this repository is a test fixture in `tests/test_biomysterybench.py`. It is adapted from preview problem `hb002` (© 2026 Anthropic, PBC, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), and its question, rubric and allowed domains are shortened. No biological data is included.

## Building the sandbox images

Each problem's data is built into its own image ahead of time, so at run time the model finds the files already extracted under `data/` and nothing is downloaded into the sandbox. Two Dockerfiles in `src/biomysterybench/` define the images:

- `Dockerfile.base` holds the shared toolset: miniforge with samtools, bcftools, bedtools, bwa, bowtie2, minimap2, STAR, salmon, subread, BLAST+, HMMER, MAFFT, seqkit and fastqc, plus biopython, pysam, pandas, numpy, scipy and scikit-learn. The model can install anything else at run time with conda or pip.
- `Dockerfile` builds one image per problem from the base and copies that problem's data under `data/`.

`build_and_push.sh` builds and pushes the base image and then one image per problem. Log in to your registry first, then run it from the repository root:

```bash
docker login registry.example.com

# full set
HF_TOKEN=hf_... \
BP_PREFIX=registry.example.com/biomystery: \
./src/biomysterybench/build_and_push.sh

# preview set, no token needed
BP_SOURCE=preview \
BP_PREFIX=registry.example.com/biomystery: \
./src/biomysterybench/build_and_push.sh
```

The image for problem `<id>` is tagged `<BP_PREFIX><id>`, and the base image is tagged `<BP_PREFIX>base` unless `BP_BASE_TAG` is set. The script takes these options as environment variables:

- `BP_SOURCE=preview` builds the 5 preview problems. The default is `full`. Any other value stops the script before it builds anything.
- `BP_IDS="hb002 hb020"` builds only the listed problems.
- `BP_MAX_GB=15` skips problems whose archive is larger than the limit.
- `BP_SKIP_EXISTING=1` skips problems whose image is already in the registry, which lets an interrupted build resume.
- `BP_PLATFORM` sets the build platform. The default is `linux/amd64`.
- `BP_BASE_TAG` sets the base image tag. The default is `<BP_PREFIX>base`.
- `BP_PRUNE_BUILD_CACHE=1` runs `docker builder prune -af` after each problem. This clears the build cache of the whole Docker host, including entries that other projects use, so it is off by default.

After each problem the script deletes the extracted data, the full-set archive in the HuggingFace cache, and the problem image in the local image store. Docker's build cache still keeps copies of each problem's data, about three times the extracted size in a local test. Without `BP_PRUNE_BUILD_CACHE=1` that cache grows with every problem, so a full build needs local disk for several times the 155 GB dataset. With it, the host needs space for the base image plus the largest single problem (about 27 GB compressed, plus the extracted data, plus the build cache for that problem). Registry storage for the pushed images is a separate cost, and the 155 GB archive total does not bound it.

The preview bundle stays in the HuggingFace cache and is reused by every problem and by later builds. If the cached bundle is corrupt, the script deletes it and downloads it again once. The cache location follows the usual HuggingFace variables `HF_HOME` and `HF_HUB_CACHE`.

To build a single problem image by hand, tag the base image as `base:latest` first, because the `Dockerfile` defaults `BASE` to `base`. The script always passes `BASE` itself, set to `BP_BASE_TAG` (default `<BP_PREFIX>base`).

## Staging data at run time

Instead of one image per problem, every sample can start from the shared base image and download its problem's data when the sample starts. Pass `-T data_source=hf` and the base image:

```bash
HF_TOKEN=hf_... uv run inspect eval biomysterybench/bio_mystery_bench \
  -T source=full -T data_source=hf \
  -T base_image=registry.example.com/biomystery@sha256:<digest> \
  --model openai/gpt-5 \
  --model-role grader=anthropic/claude-opus-4-8
```

Before the agent starts, the `stage_problem_data` setup step:

1. resolves a short-lived URL for the problem's archive in the process running Inspect. With `hf` this is the signed CDN link HuggingFace redirects to, which needs no token and expires after about an hour. With `s3` it is a presigned URL into a mirror of the archives (see below).
2. writes the URL into the sandbox with sandbox events turned off, so it is not recorded in the eval log;
3. downloads the archive to `/tmp/bmb-stage/` in the sandbox, outside the agent's working directory, and checks its size and sha256 against `manifest.json`;
4. extracts it into `/workspace/data/` exactly as `build_and_push.sh` does (the preview bundle is flattened from `<id>/`) and deletes the archive and the URL.

`HF_TOKEN` and AWS credentials never enter the sandbox. This matters because the agent has internet access: with the token it could download `problems.csv`, which holds the rubrics of the gated full set. Each attempt resolves a fresh URL. After 3 failed attempts the sample is marked ERRORED, so it is eligible for `--retry-on-error` and is not scored as a wrong answer. The task records the staging time, archive size and extracted size in the sample's `bmb_staging` metadata.

Pin the base image by digest. `Dockerfile.base` does not pin its conda packages, so a rebuild under the same tag can change the tools. The task logs a warning when the base image has no digest.

For `-T data_source=s3`, copy each archive to `s3://<bucket>/<prefix>/<repo>/<revision>/<archive path>` (for example `.../Anthropic/BioMysteryBench-full/b5a889c.../data/hb002.zip`), install the `s3` extra for `boto3`, and pass `-T s3_uri=s3://<bucket>/<prefix>` or set `BMB_S3_URI`. The URL is presigned with the AWS credentials of the process running Inspect.

`manifest.json` holds each problem's archive path, size, LFS sha256, size tier and `human_solvable` flag at the pinned revisions, and no problem text. Regenerate it with `HF_TOKEN=hf_... uv run python scripts/make_manifest.py` whenever a revision changes. The task refuses to run if the manifest's revision differs from the pinned one.

## Running

Set the same prefix you built with, then run the task with a grader model:

```bash
export BMB_IMAGE_PREFIX=registry.example.com/biomystery:

# 5 open preview problems, no token needed
uv run inspect eval biomysterybench/bio_mystery_bench \
  -T source=preview \
  --model openai/gpt-5 \
  --model-role grader=anthropic/claude-opus-4-8

# full 90-problem set, 5 trials per problem
HF_TOKEN=hf_... uv run inspect eval biomysterybench/bio_mystery_bench \
  -T source=full -T epochs=5 \
  --model openai/gpt-5 \
  --model-role grader=anthropic/claude-opus-4-8
```

The default sandbox is Inspect's `docker` provider, configured by `src/biomysterybench/compose.yaml`. To run on Kubernetes, install the `k8s` extra (see [Installation](#installation)), configure `kubectl` access to your cluster as the [inspect_k8s_sandbox documentation](https://k8s-sandbox.aisi.org.uk/) describes, and pass `--sandbox k8s:src/biomysterybench/compose.yaml`. The k8s provider converts the compose file and sets both the request and the limit of each pod to the compose values.

Anthropic reports mean accuracy over 5 one-shot trials on the 73 problems that at least one human benchmarker solved. Pass `-T human_solvable_only=true` to restrict a run to that subset.

### Task arguments

- `source` (`str`): `"full"` (90 gated problems) or `"preview"` (5 open problems). Default `"full"`.
- `ids` (`str | list[str] | None`): run only these problem ids. Accepts a list or a single id. Default `None`.
- `limit` (`int | None`): keep only the first N problems after filtering. Must be zero or more. Default `None`.
- `human_solvable_only` (`bool`): keep only problems that at least one human solved. Default `False`.
- `tier` (`str | list[str] | None`): keep only problems whose archive is in these size tiers: `"small"` (under 2 GB, 76 problems), `"medium"` (2 to 10 GB, 8) or `"large"` (10 GB or more, 6). Read from `manifest.json`. Default `None`.
- `data_source` (`str`): `"image"` uses the per-problem images. `"hf"` or `"s3"` stage the data at run time (see [Staging data at run time](#staging-data-at-run-time)). Default `"image"`.
- `image_prefix` (`str | None`): per-problem image prefix. A sample's image is `image_prefix + problem_id`. Defaults to the `BMB_IMAGE_PREFIX` environment variable. Used only with `data_source="image"`.
- `base_image` (`str | None`): the shared base image for `data_source` `"hf"` or `"s3"`, ideally pinned by digest. Defaults to the `BMB_BASE_IMAGE` environment variable.
- `s3_uri` (`str | None`): `s3://bucket/prefix` of the archive mirror for `data_source="s3"`. Defaults to the `BMB_S3_URI` environment variable.
- `sandbox_memory` (`str`): memory for each sandbox as a docker compose byte value such as `"32G"` or `"512M"`. On Kubernetes this is both the memory request and the memory limit of the pod. Default `"32G"`.
- `sandbox_disk` (`str | None`): ephemeral storage for each sandbox as a Kubernetes quantity such as `"50Gi"`, set as both the request and the limit of the pod through `disk-compose.yaml`. Kubernetes schedules the pod only on a node with that much free disk and evicts it if it writes more. Docker ignores it. Default `None`, which sets no disk request.
- `stage_timeout` (`int`): seconds allowed for one download and extraction when staging. Default `7200`.
- `with_sandbox` (`bool`): attach the sandbox. `False` builds the task without one, for offline tests. Default `True`.
- `epochs` (`int`): trials per problem. Default `1`.
- `tool_timeout` (`int`): timeout for each bash or python call in seconds. Default `600`.
- `message_limit` (`int | None`): maximum number of messages in a sample, shared by all its submissions. Default `150`.
- `token_limit` (`int | None`): optional limit on the agent's tokens in a sample. Grader tokens do not count toward it. Default `None`.
- `max_tool_output` (`int | None`): truncate each tool result to this many bytes. `None` uses Inspect's default. Default `None`.
- `max_attempts` (`int`): submissions the agent may make per sample (see [Multiple submissions](#multiple-submissions)). `1` is upstream's one-shot protocol. Must be 1 or more. Default `1`.

### Multiple submissions

By default the agent submits once, as upstream. With `-T max_attempts=5` it may submit up to 5 times. Each submission except the last is graded when it is made, by the same grader as the final score. A correct submission ends the sample. After an incorrect one the agent is told only "Your submission was incorrect. Please proceed and attempt to find the correct answer.", never the grader's reasoning or the rubric, and it keeps working. The last submission is graded at the end of the sample, as in a one-shot run. The system prompt does not tell the agent how many submissions it has. This is the mechanism of LabBench2's `agentic_max_attempts`.

`message_limit` and `token_limit` are shared by all the submissions. Each incorrect submission uses 3 messages: the `submit` call, its result and the reply.

`max_attempts` and `epochs` are independent:

| Protocol | Task arguments |
|---|---|
| One-shot, one trial (default) | `-T max_attempts=1 -T epochs=1` |
| Up to 5 submissions in one trial | `-T max_attempts=5 -T epochs=1` |
| pass@5: 5 independent one-shot trials | `-T max_attempts=1 -T epochs=5` |
| Both | `-T max_attempts=5 -T epochs=5` |

Each epoch keeps its own attempt record. The epoch reducer is `mean`, and every epoch's score is kept in the log, so pass@k is computed from those.

## Scoring

The scorer `biomystery_grader` is built on Inspect's `model_graded_qa`. The judge receives the full transcript, the submitted answer and the problem's `answer_rubric`. It returns `GRADE: C` only when the answer matches the rubric and the model did not cheat. Cheating means identifying the source dataset by looking up a GEO, SRA, ENA or BioProject accession, the source publication or the study metadata. Standard database use such as gene or sequence lookup, reference genome download and BLAST is allowed.

The judge comes from the `grader` model role. If the role is not set, grading falls back to the model under test, and that model grades its own answers. Runs to date used `anthropic/claude-opus-4-8` as the grader.

A sample that the judge did not grade is left unscored and excluded from accuracy. This happens when the judge's reply contains no `GRADE:` line. Two causes have been seen or are possible. A Claude grader's safety classifier can stop the grader on biological content with `stop_reason="content_filter"` and an empty reply. A transcript longer than the grader's context window ends with `stop_reason="model_length"`. Inspect AI 0.3.245 and later record these samples as unscored and report their number as `unscored_samples` in the log's results for the scorer. The reason is recorded in `Score.reason` on recent releases and in `metadata["unscored_reason"]` on older ones. Earlier versions score them INCORRECT, which is why the package requires 0.3.245. Before leaving a sample unscored, the scorer asks the judge again, up to 3 calls in all. A `content_filter` stop can clear on a retry.

With more than one submission allowed, the judge is also told that any earlier submission in the transcript was judged incorrect, that it should grade only the final submitted answer, and that the cheating check covers the whole transcript. With `max_attempts=1` the judge's instructions are the same as in 2-B.

If the sample ends on a limit or a context overflow after the agent has submitted, without a new submission, the judge grades the last submission rather than the model's last text. The logged output and transcript are left as they were.

Each score's metadata also records:

- `submitted`, which is `True` only when the agent called the `submit` tool.
- `n_submissions`, the number of `submit` calls.
- `attempt_grades`, the grades given during the sample, in order, followed by the final grade. An unscored grade is `null`. A submission graded correct during the sample is graded again at the end, so it appears twice.
- `first_attempt_correct`, whether the first graded answer was correct, for comparison with one-shot runs. `null` when that grade was unscored.
- `graded_last_submission`, `True` when the last submission was graded in place of the model's last text (see above).
- `grader_attempts`, the number of judge calls behind this grade.
- `problem_id` and `human_solvable`.
- `stop_reason`, the stop reason of the model's last generation. It describes that one generation, not why the sample ended.
- `sandbox_resources`, read from the sandbox at scoring time: `memory_peak_bytes` (cgroup `memory.peak`, which includes page cache, so it is an upper bound on what the agent's processes needed), `oom_kills` (processes the kernel killed at the memory limit) and `workspace_bytes` (`du` of `/workspace`). A value the sandbox does not expose is `null`. These are for sizing sandboxes.

To find samples that ran out of messages or tokens, read each sample's `limit` field in the log. Inspect sets it to the limit that ended the sample (for example `type="message"`), and leaves it empty for a sample that finished normally.

## Known limitations

- **`allowed_domains` is not enforced.** It is carried into sample metadata, but the sandbox has unrestricted network access. The rule against looking up the source study rests on the system prompt and on the judge noticing a lookup in the transcript.
- **Some problems may never be graded by a Claude judge.** In one run of 90 problems with 5 epochs, 9 of 450 grader calls to `claude-opus-4-8` ended with `content_filter`. All 5 epochs of `hb019` were affected. Check `unscored_samples` in each log, and treat a problem with every epoch unscored as missing from the results, not as a zero.
- **Image storage is large.** The full set needs about 155 GB of registry storage, and problems with large datasets may need more than the default 32 GB of sandbox memory.
- **Execution environment.** Runs to date used Kubernetes through inspect_k8s_sandbox. The local Docker path is configured but has not been run end to end on the full set. Local Docker needs a host with at least 4 CPUs unless `cpus` is lowered in `compose.yaml`.
- **Unsubmitted samples are still graded.** A sample that hits `message_limit` or `token_limit` before calling `submit` goes to the grader like any other. In runs to date these samples scored INCORRECT. They are identified by `submitted: false` in the score metadata and by the sample's `limit` field in the log.
- **Grading during the sample adds cost and time.** Up to 4 extra judge calls per sample, each with the full transcript, while the sandbox is still held. Since grader tokens do not count toward `token_limit`, the limit does not bound a sample's total tokens. If a judge call fails with an API error during the sample, the sample errors and is rerun from the start under `--retry-on-error`.

## Future work

- **Pin the conda packages in `Dockerfile.base`.** The Python packages are pinned but the conda and bioconda tools are not, so two builds of the base image at different times can differ. The dataset revisions are pinned, but the execution environment is not reproducible.
- **Label images with their provenance.** `BP_SKIP_EXISTING=1` checks only that a tag exists. An image built from another dataset revision, platform or base would be reused silently. Recording the dataset revision, platform and base digest as image labels, or in the tag, would make resumed builds auditable.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src tests
```

Most tests run offline. They mock the HuggingFace download and the models and do not start a sandbox. The tests in `tests/test_install.py` install the package into fresh virtual environments, once as an editable install and once from a built wheel, and check that Inspect resolves `biomysterybench/bio_mystery_bench` in a new process. They need `uv` on `PATH` and may download packages that are not in the uv cache. Skip them with `uv run pytest -m "not install"`.

## Licence and attribution

The code is released under the MIT licence (see [LICENSE](LICENSE)). It was first written in a fork of the UK AI Security Institute's [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) and follows that repository's structure and conventions. The licence keeps the UK AI Security Institute's copyright notice for that reason.

BioMysteryBench was created by Anthropic. Cite Anthropic's [research page](https://www.anthropic.com/research/Evaluating-Claude-For-Bioinformatics-With-BioMysteryBench) and the HuggingFace datasets when publishing results.

## Changelog

### [3-B] - 2026-10-28

- Add the `max_attempts` task argument. The default, `1`, keeps the one-shot protocol. With `max_attempts=5` the agent may submit up to 5 times and is told only that a wrong submission was incorrect (see [Multiple submissions](#multiple-submissions)).
- With more than one submission allowed, the judge's instructions say that earlier submissions were judged incorrect.
- After a limit or context overflow that follows a submission, the last submission is graded instead of the model's last text.
- The judge is asked up to 3 times before a sample is left unscored.
- Grader tokens no longer count toward `token_limit`.
- Record `n_submissions`, `attempt_grades`, `first_attempt_correct`, `graded_last_submission` and `grader_attempts` in the score metadata, and `max_attempts` in the task metadata.
- A default (one-shot) run differs from 2-B only through the judge retry, which can score a sample that 2-B left unscored, so the comparability number changes. Runs with `max_attempts` above 1 are a different protocol and are not comparable with any one-shot run.
- Add run-time data staging: the `data_source`, `base_image`, `s3_uri` and `stage_timeout` task arguments, the `stage_problem_data` setup step and `manifest.json`. The default, `data_source="image"`, behaves as before. Staged data was checked file by file (sha256, mode and owner) against the per-problem images for the 5 preview problems and 5 full-set problems.
- Add the `tier` and `sandbox_disk` task arguments, and `disk-compose.yaml` (named so that it ends in `compose.yaml`, which Kubernetes conversion requires).
- Record `sandbox_resources` in the score metadata, and `data_source`, `base_image`, the dataset revision and the harness commit in the task metadata.

### [2-B] - 2026-09-24

- Add the `sandbox_memory` task argument (default `32G`). It sets `mem_limit` in `compose.yaml`.
- Set `cpus: 4` in `compose.yaml`.
- On Kubernetes each value is both the request and the limit of the sandbox pod.
- Results may not be comparable with 1-A. Under 1-A, `compose.yaml` set no memory or CPU limits.

### [1-A] - 2026-08-10

Initial version.
