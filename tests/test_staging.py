"""Offline tests for run-time data staging, the manifest, tiers and telemetry.

The staging script itself is run for real against a temporary directory, with a
file:// URL in place of the signed link, so its download, checks and extraction
are exercised without a sandbox or the network.
"""

import asyncio
import contextlib
import hashlib
import io
import shutil
import subprocess
import urllib.error
import zipfile
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import biomysterybench.biomysterybench as bmb
import biomysterybench.scorer as scoring
import biomysterybench.staging as staging
from biomysterybench import bio_mystery_bench
from biomysterybench.manifest import TIERS, ProblemArchive, load_source


def _archive(**overrides) -> ProblemArchive:
    fields = dict(
        problem_id="hb002",
        repo="Anthropic/BioMysteryBench-full",
        revision=bmb.FULL_REVISION,
        archive="data/hb002.zip",
        size=4,
        sha256="0" * 64,
        tier="small",
        human_solvable=True,
    )
    return ProblemArchive(**{**fields, **overrides})


# --- manifest ----------------------------------------------------------------


@pytest.mark.parametrize(
    "source,repo,revision,count",
    [
        ("full", bmb.FULL_REPO, bmb.FULL_REVISION, 90),
        ("preview", bmb.PREVIEW_REPO, bmb.PREVIEW_REVISION, 5),
    ],
)
def test_manifest_matches_pinned_revision(source, repo, revision, count):
    archives = load_source(source, repo, revision)
    assert len(archives) == count
    for pid, a in archives.items():
        assert a.problem_id == pid
        assert len(a.sha256) == 64 and int(a.sha256, 16) >= 0
        assert a.size > 0
        assert a.tier in TIERS
    if source == "preview":
        assert {a.archive for a in archives.values()} == {"data.zip"}
        assert all(a.bundled for a in archives.values())
    else:
        assert all(a.archive == f"data/{pid}.zip" for pid, a in archives.items())


def test_manifest_tiers_follow_archive_size():
    archives = load_source("full", bmb.FULL_REPO, bmb.FULL_REVISION).values()
    for a in archives:
        expected = "small" if a.size < 2e9 else "medium" if a.size < 10e9 else "large"
        assert a.tier == expected
    counts = {t: sum(a.tier == t for a in archives) for t in TIERS}
    assert counts == {"small": 76, "medium": 8, "large": 6}
    assert sum(a.human_solvable for a in archives) == 73


def test_manifest_refuses_another_revision():
    with pytest.raises(ValueError, match="regenerate it"):
        load_source("full", bmb.FULL_REPO, "0" * 40)


# --- URL resolution ----------------------------------------------------------


def _redirect(code, location):
    headers = Message()
    headers["Location"] = location
    return urllib.error.HTTPError("u", code, "redirect", headers, io.BytesIO())


def _fake_opener(monkeypatch, responses):
    seen = []

    class Opener:
        def open(self, request, timeout):
            seen.append(request)
            raise responses.pop(0)

    monkeypatch.setattr(staging.urllib.request, "build_opener", lambda *a: Opener())
    return seen


def test_hf_url_is_the_cdn_redirect(monkeypatch):
    seen = _fake_opener(monkeypatch, [_redirect(302, "https://cdn.example/signed")])
    url = staging.resolve_hf_url(_archive(), "hf_secret")
    assert url == "https://cdn.example/signed"
    [request] = seen
    assert request.get_method() == "HEAD"
    assert request.full_url == (
        "https://huggingface.co/datasets/Anthropic/BioMysteryBench-full/resolve/"
        f"{bmb.FULL_REVISION}/data/hb002.zip"
    )
    assert request.get_header("Authorization") == "Bearer hf_secret"


def test_hf_url_follows_relative_redirects_with_the_token(monkeypatch):
    seen = _fake_opener(
        monkeypatch,
        [_redirect(307, "/datasets/new/resolve/x"), _redirect(302, "https://cdn/s")],
    )
    assert staging.resolve_hf_url(_archive(), "t") == "https://cdn/s"
    assert seen[1].full_url == "https://huggingface.co/datasets/new/resolve/x"
    assert seen[1].get_header("Authorization") == "Bearer t"


def test_hf_url_reports_access_errors(monkeypatch):
    _fake_opener(
        monkeypatch,
        [urllib.error.HTTPError("u", 401, "no", Message(), io.BytesIO())],
    )
    with pytest.raises(RuntimeError, match="HTTP 401.*HF_TOKEN"):
        staging.resolve_hf_url(_archive(), None)


def test_s3_key_includes_repo_and_revision():
    key = staging.s3_key("/mirror/", _archive())
    assert (
        key
        == f"mirror/Anthropic/BioMysteryBench-full/{bmb.FULL_REVISION}/data/hb002.zip"
    )
    assert staging.s3_key("", _archive()).startswith("Anthropic/")


def test_s3_needs_a_uri(monkeypatch):
    monkeypatch.delenv(staging.S3_URI_ENV, raising=False)
    with pytest.raises(ValueError, match="s3_uri"):
        staging.resolve_url("s3", _archive())
    with pytest.raises(ValueError, match="s3://bucket"):
        staging.resolve_s3_url(_archive(), "https://bucket/prefix")


# --- stage_problem_data (sandbox mocked) -------------------------------------


class FakeSandbox:
    """Records calls; exec results come from `results` in order."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self.events = True

    async def exec(self, cmd, timeout=None):
        self.calls.append(("exec", cmd, self.events))
        if cmd[0] == "sh":
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return SimpleNamespace(success=True, stdout="", stderr="")

    async def write_file(self, file, contents):
        self.calls.append(("write_file", file, contents, self.events))

    @contextlib.contextmanager
    def no_events(self):
        self.events = False
        try:
            yield
        finally:
            self.events = True


def _ok(stdout="1234\n"):
    return SimpleNamespace(success=True, stdout=stdout, stderr="")


def _fail(stderr="curl: (22) 403"):
    return SimpleNamespace(success=False, stdout="", stderr=stderr)


def _stage(monkeypatch, sb, archive=None, urls=None, **kwargs):
    archive = archive or _archive()
    urls = list(urls or ["https://cdn/signed?X-Sig=secret"] * 5)

    def fake_resolve(source, a, s3_uri=None):
        url = urls.pop(0)
        if isinstance(url, BaseException):
            raise url
        return url

    monkeypatch.setattr(staging, "sandbox", lambda: sb)
    monkeypatch.setattr(staging, "resolve_url", fake_resolve)
    state = SimpleNamespace(sample_id="hb002", metadata={"problem_id": "hb002"})
    solve = staging.stage_problem_data({"hb002": archive}, "hf", **kwargs)
    return asyncio.run(solve(state, None)), state


def test_stage_hides_the_url_and_records_staging(monkeypatch):
    sb = FakeSandbox([_ok("4567\n")])
    _, state = _stage(monkeypatch, sb)

    [write] = [c for c in sb.calls if c[0] == "write_file"]
    assert write[1] == staging._URL_FILE
    assert write[3] is False, "the URL must be written with sandbox events off"
    for call in sb.calls:
        if call[0] == "exec":
            assert call[2] is True
            assert not any("secret" in part for part in call[1])
    [script] = [c[1] for c in sb.calls if c[0] == "exec" and c[1][0] == "sh"]
    assert script[3:] == ["stage", "hb002", "0" * 64, "4", "single"]

    meta = state.metadata["bmb_staging"]
    assert meta["data_bytes"] == 4567 and meta["attempts"] == 1
    assert meta["archive"] == "data/hb002.zip"


def test_stage_marks_the_preview_bundle(monkeypatch):
    sb = FakeSandbox([_ok()])
    _stage(monkeypatch, sb, archive=_archive(archive="data.zip"))
    [script] = [c[1] for c in sb.calls if c[0] == "exec" and c[1][0] == "sh"]
    assert script[-1] == "bundle"


def test_stage_retries_with_a_fresh_url_then_succeeds(monkeypatch):
    sb = FakeSandbox([_fail(), TimeoutError(), _ok()])
    urls = ["https://a", "https://b", "https://c"]
    _, state = _stage(monkeypatch, sb, urls=urls)
    written = [c[2] for c in sb.calls if c[0] == "write_file"]
    assert written == ["https://a", "https://b", "https://c"]
    assert state.metadata["bmb_staging"]["attempts"] == 3
    cleanups = [c for c in sb.calls if c[0] == "exec" and c[1][0] == "rm"]
    assert len(cleanups) == 2


def test_stage_errors_the_sample_after_every_attempt_fails(monkeypatch):
    sb = FakeSandbox([_fail("archive sha256 is 1f, manifest says 00")] * 2)
    urls = [RuntimeError("HTTP 429"), "https://a", "https://b"]
    with pytest.raises(RuntimeError, match="could not stage") as e:
        _stage(monkeypatch, sb, urls=urls, attempts=3)
    assert "HTTP 429" in str(e.value) and "manifest says" in str(e.value)


# --- the staging script, run for real ----------------------------------------

needs_tools = pytest.mark.skipif(
    not all(shutil.which(t) for t in ("curl", "sha256sum", "python3", "stat")),
    reason="needs curl, sha256sum, stat and python3",
)


def _run_script(tmp_path, archive_bytes, pid, mode, sha=None, size=None, data=None):
    workspace = tmp_path / "workspace"
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    workspace.mkdir()
    if data:
        (workspace / "data").mkdir()
        (workspace / "data" / "old").write_text(data)
    src = tmp_path / "remote.zip"
    src.write_bytes(archive_bytes)
    (stage_dir / "url").write_text(src.as_uri())
    script = staging._STAGE_SCRIPT.replace(staging.STAGE_DIR, str(stage_dir)).replace(
        "/workspace", str(workspace)
    )
    result = subprocess.run(
        [
            "sh",
            "-c",
            script,
            "stage",
            pid,
            sha or hashlib.sha256(archive_bytes).hexdigest(),
            str(size if size is not None else len(archive_bytes)),
            mode,
        ],
        capture_output=True,
        text=True,
    )
    return result, workspace / "data", stage_dir


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


@needs_tools
def test_script_extracts_a_single_archive(tmp_path):
    archive = _zip({"reads.fastq": "@r1\nACGT\n", "sub/ref.fa": ">x\nAC\n"})
    result, data, stage_dir = _run_script(tmp_path, archive, "hb002", "single")
    assert result.returncode == 0, result.stderr
    assert (data / "reads.fastq").read_text() == "@r1\nACGT\n"
    assert (data / "sub" / "ref.fa").exists()
    assert list(stage_dir.iterdir()) == [], "URL and archive must be deleted"
    assert int(result.stdout.split()[-1]) > 0


@needs_tools
def test_script_flattens_one_problem_from_the_bundle(tmp_path):
    archive = _zip({"hb002/a.txt": "a", "hb002/d/b.txt": "b", "hb020/c.txt": "c"})
    result, data, _ = _run_script(tmp_path, archive, "hb002", "bundle")
    assert result.returncode == 0, result.stderr
    assert sorted(p.relative_to(data).as_posix() for p in data.rglob("*")) == [
        "a.txt",
        "d",
        "d/b.txt",
    ]


@needs_tools
@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"sha": "0" * 64}, "manifest says 000"),
        ({"size": 1}, "manifest says 1"),
        ({"data": "left over"}, "not empty"),
    ],
)
def test_script_rejects_bad_archives(tmp_path, kwargs, message):
    archive = _zip({"a.txt": "a"})
    result, _, stage_dir = _run_script(tmp_path, archive, "hb002", "single", **kwargs)
    assert result.returncode != 0
    assert message in result.stderr
    assert list(stage_dir.iterdir()) == []


@needs_tools
def test_script_rejects_a_problem_missing_from_the_bundle(tmp_path):
    result, _, _ = _run_script(tmp_path, _zip({"hb020/c": "c"}), "hb002", "bundle")
    assert result.returncode != 0 and "not in the bundle" in result.stderr


def test_script_never_names_the_url():
    assert "$(cat " + staging._URL_FILE + ")" in staging._STAGE_SCRIPT


# --- task wiring -------------------------------------------------------------

_ROWS = [
    {"id": pid, "question": "q", "answer_rubric": "r", "human_solvable": hs}
    for pid, hs in [
        ("hb002", "yes"),
        ("reccwgc4buredxvyz", "yes"),
        ("hb008", "yes"),
    ]
]
_BASE = "ghcr.io/example/biomystery-base@sha256:" + "a" * 64


@pytest.fixture
def full_rows(monkeypatch):
    monkeypatch.setattr(bmb, "_load_problems", lambda *a, **k: _ROWS)


def test_hf_source_uses_the_base_image_and_stages(full_rows):
    task = bio_mystery_bench(data_source="hf", base_image=_BASE)
    assert {s.metadata["BMB_IMAGE"] for s in task.dataset} == {_BASE}
    assert task.metadata["data_source"] == "hf"
    assert task.metadata["base_image"] == _BASE
    assert task.metadata["dataset"].endswith(bmb.FULL_REVISION)
    sample = next(s for s in task.dataset if s.id == "hb002")
    assert sample.metadata["tier"] == "small"
    assert len(sample.metadata["archive_sha256"]) == 64


def test_image_source_keeps_per_problem_images(full_rows):
    task = bio_mystery_bench(image_prefix="repo:")
    assert {s.metadata["BMB_IMAGE"] for s in task.dataset} == {
        "repo:hb002",
        "repo:reccwgc4buredxvyz",
        "repo:hb008",
    }
    assert task.metadata["data_source"] == "image"


def test_staged_sources_need_a_base_image(full_rows, monkeypatch):
    monkeypatch.delenv("BMB_BASE_IMAGE", raising=False)
    with pytest.raises(ValueError, match="base_image is unset"):
        bio_mystery_bench(data_source="hf")
    # The base image replaces the per-problem prefix, which is then not needed.
    monkeypatch.delenv("BMB_IMAGE_PREFIX", raising=False)
    bio_mystery_bench(data_source="hf", base_image=_BASE)


def test_unpinned_base_image_warns(full_rows, caplog):
    bio_mystery_bench(data_source="hf", base_image="ghcr.io/example/base:latest")
    assert "not pinned by digest" in caplog.text


def test_rejects_unknown_data_source_and_tier(full_rows):
    with pytest.raises(ValueError, match="data_source"):
        bio_mystery_bench(data_source="ftp", with_sandbox=False)
    with pytest.raises(ValueError, match="tier must be"):
        bio_mystery_bench(tier="huge", with_sandbox=False)


@pytest.mark.parametrize(
    "tier,expected",
    [
        ("small", ["hb002"]),
        ("large", ["reccwgc4buredxvyz"]),
        (["medium", "large"], ["reccwgc4buredxvyz", "hb008"]),
    ],
)
def test_tier_selects_problems(full_rows, tier, expected):
    task = bio_mystery_bench(tier=tier, with_sandbox=False)
    assert [s.id for s in task.dataset] == expected


def test_csv_ids_missing_from_manifest_fail(monkeypatch):
    rows = [{"id": "hb999", "question": "q", "answer_rubric": "r"}]
    monkeypatch.setattr(bmb, "_load_problems", lambda *a, **k: rows)
    with pytest.raises(ValueError, match="missing from manifest"):
        bio_mystery_bench(with_sandbox=False)


def _render(compose_path, sample):
    text = Path(compose_path).read_text()
    for key, value in sample.metadata.items():
        text = text.replace(f"${{SAMPLE_METADATA_{key.upper()}}}", str(value))
    assert "${" not in text
    return yaml.safe_load(text)["services"]["default"]


def test_sandbox_disk_requests_ephemeral_storage(full_rows):
    task = bio_mystery_bench(image_prefix="repo:", sandbox_disk="50Gi")
    assert task.sandbox is not None and task.sandbox.config == bmb._COMPOSE_DISK
    service = _render(bmb._COMPOSE_DISK, next(iter(task.dataset)))
    resources = service["x-inspect_k8s_sandbox"]["resources"]
    assert resources == {
        "requests": {"ephemeral-storage": "50Gi"},
        "limits": {"ephemeral-storage": "50Gi"},
    }


def test_no_sandbox_disk_keeps_upstream_compose(full_rows):
    task = bio_mystery_bench(image_prefix="repo:")
    assert task.sandbox is not None and task.sandbox.config == bmb._COMPOSE
    assert "BMB_DISK" not in next(iter(task.dataset)).metadata


def test_disk_compose_differs_only_by_the_disk_block():
    plain = yaml.safe_load(Path(bmb._COMPOSE).read_text())
    disk = yaml.safe_load(Path(bmb._COMPOSE_DISK).read_text())
    disk["services"]["default"].pop("x-inspect_k8s_sandbox")
    assert disk == plain


@pytest.mark.parametrize("value", ["50GB", "lots", "", "50 Gi"])
def test_rejects_bad_sandbox_disk(value):
    with pytest.raises(ValueError, match="sandbox_disk"):
        bio_mystery_bench(image_prefix="repo:", sandbox_disk=value)


# --- telemetry ---------------------------------------------------------------


def _telemetry_sandbox(monkeypatch, stdout=None, error=None):
    async def fake_exec(cmd, timeout=None):
        if error:
            raise error
        return SimpleNamespace(success=True, stdout=stdout, stderr="")

    monkeypatch.setattr(scoring, "sandbox", lambda: SimpleNamespace(exec=fake_exec))


def test_telemetry_parses_cgroup_and_disk(monkeypatch):
    _telemetry_sandbox(monkeypatch, "8589934592 1 123456\n")
    assert asyncio.run(scoring.sandbox_resources()) == {
        "memory_peak_bytes": 8589934592,
        "oom_kills": 1,
        "workspace_bytes": 123456,
    }


def test_telemetry_marks_missing_values(monkeypatch):
    _telemetry_sandbox(monkeypatch, "- - 99\n")
    assert asyncio.run(scoring.sandbox_resources()) == {
        "memory_peak_bytes": None,
        "oom_kills": None,
        "workspace_bytes": 99,
    }


def test_telemetry_survives_a_missing_sandbox(monkeypatch):
    _telemetry_sandbox(monkeypatch, error=ProcessLookupError("no sandbox"))
    assert asyncio.run(scoring.sandbox_resources()) is None
