"""Stage a problem's data into a shared base-image sandbox at sample start.

With `data_source="image"` (the default) each problem's data is baked into its
own image and nothing here runs. With `"hf"` or `"s3"` every sample starts from
one shared base image, and `stage_problem_data` downloads the problem's archive
into the sandbox and extracts it under `data/`, exactly as `build_and_push.sh`
lays it out in a per-problem image.

Credentials never enter the sandbox. The runner resolves a short-lived URL that
needs no token (a signed HuggingFace CDN link, or a presigned S3 URL) and hands
the sandbox only that. The URL is written with sandbox events off, so it is not
recorded in the eval log, and is deleted once the download finishes.
"""

import asyncio
import contextlib
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import SandboxEnvironment, sandbox

from biomysterybench.manifest import ProblemArchive

logger = logging.getLogger(__name__)

DATA_SOURCES = ("image", "hf", "s3")

_HF_ENDPOINT_ENV = "HF_ENDPOINT"
_HF_ENDPOINT = "https://huggingface.co"
S3_URI_ENV = "BMB_S3_URI"
_URL_TTL_S = 3600

# Outside /workspace, so the agent never sees the archive.
STAGE_DIR = "/tmp/bmb-stage"
_URL_FILE = f"{STAGE_DIR}/url"
_ARCHIVE = f"{STAGE_DIR}/archive.zip"

# Runs in the sandbox. The URL is read from a file, so the logged command line
# never contains it. $1 problem id, $2 archive sha256, $3 archive size in bytes,
# $4 "bundle" for the preview bundle or "single".
_STAGE_SCRIPT = f"""
set -eu
trap 'rm -f {_URL_FILE} {_ARCHIVE}' EXIT
if [ -e /workspace/data ] && [ -n "$(ls -A /workspace/data)" ]; then
    echo "/workspace/data is not empty before staging" >&2; exit 1
fi
curl -fsS --retry 5 --retry-all-errors --connect-timeout 30 \\
    -o {_ARCHIVE} "$(cat {_URL_FILE})"
rm -f {_URL_FILE}
size=$(stat -c %s {_ARCHIVE})
if [ "$size" != "$3" ]; then
    echo "archive is $size bytes, manifest says $3" >&2; exit 1
fi
sha=$(sha256sum {_ARCHIVE} | cut -d' ' -f1)
if [ "$sha" != "$2" ]; then
    echo "archive sha256 is $sha, manifest says $2" >&2; exit 1
fi
python3 - {_ARCHIVE} /workspace/data "$1" "$4" <<'PY'
# Mirrors the extraction in build_and_push.sh, whose output Dockerfile copies
# to /workspace/data/.
import os, sys, zipfile
archive, dest, pid, mode = sys.argv[1:5]
os.makedirs(dest, exist_ok=True)
with zipfile.ZipFile(archive) as z:
    if mode == "bundle":
        members = [m for m in z.namelist() if m.startswith(pid + "/")]
        if not members:
            sys.exit(pid + " not in the bundle")
        for m in members:
            z.extract(m, dest)
        inner = os.path.join(dest, pid)
        for name in os.listdir(inner):
            os.rename(os.path.join(inner, name), os.path.join(dest, name))
        os.rmdir(inner)
    else:
        z.extractall(dest)
PY
du -sb /workspace/data | cut -f1
"""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def resolve_hf_url(archive: ProblemArchive, token: str | None) -> str:
    """Return a signed CDN link for the archive that works without a token.

    HuggingFace answers a `resolve` request for an LFS file with a redirect to a
    signed link that expires after about an hour. Relative redirects (a renamed
    repo) are followed with the token; the first absolute one is the link.
    """
    endpoint = os.environ.get(_HF_ENDPOINT_ENV, _HF_ENDPOINT).rstrip("/")
    url = (
        f"{endpoint}/datasets/{archive.repo}/resolve/{archive.revision}/"
        f"{urllib.parse.quote(archive.archive)}"
    )
    opener = urllib.request.build_opener(_NoRedirect)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for _ in range(5):
        request = urllib.request.Request(url, method="HEAD", headers=headers)
        try:
            with opener.open(request, timeout=60) as response:
                raise RuntimeError(
                    f"{archive.archive}: expected a redirect to the CDN, got "
                    f"HTTP {response.status}"
                )
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308):
                raise RuntimeError(
                    f"{archive.repo}/{archive.archive}: HTTP {e.code} from "
                    "HuggingFace. Check that HF_TOKEN is set and has access."
                ) from e
            location = e.headers["Location"]
        if location.startswith("/"):
            url = endpoint + location
            continue
        return location
    raise RuntimeError(f"{archive.archive}: too many redirects from HuggingFace")


def s3_key(prefix: str, archive: ProblemArchive) -> str:
    """Object key of a mirrored archive: <prefix>/<repo>/<revision>/<archive>.

    The revision is part of the key, so a mirror of another revision is never
    read by mistake.
    """
    parts = [prefix.strip("/"), archive.repo, archive.revision, archive.archive]
    return "/".join(p for p in parts if p)


def resolve_s3_url(archive: ProblemArchive, s3_uri: str) -> str:
    """Presign a GET for the mirrored archive with the runner's AWS credentials."""
    parsed = urllib.parse.urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"s3_uri must look like s3://bucket/prefix, got {s3_uri!r}")
    try:
        import boto3  # type: ignore[import-not-found,import-untyped,unused-ignore]
    except ImportError as e:
        raise RuntimeError(
            "data_source='s3' needs boto3; install biomysterybench[s3]"
        ) from e
    client = boto3.client("s3")
    return str(
        client.generate_presigned_url(
            "get_object",
            Params={"Bucket": parsed.netloc, "Key": s3_key(parsed.path, archive)},
            ExpiresIn=_URL_TTL_S,
        )
    )


def resolve_url(
    data_source: str, archive: ProblemArchive, s3_uri: str | None = None
) -> str:
    """Return a short-lived, token-free URL for the archive."""
    if data_source == "hf":
        return resolve_hf_url(archive, os.environ.get("HF_TOKEN") or None)
    if data_source == "s3":
        uri = s3_uri or os.environ.get(S3_URI_ENV)
        if not uri:
            raise ValueError(f"data_source='s3' needs s3_uri or {S3_URI_ENV}")
        return resolve_s3_url(archive, uri)
    raise ValueError(f"no URL to resolve for data_source={data_source!r}")


@contextlib.contextmanager
def _without_events(sb: SandboxEnvironment) -> Iterator[None]:
    # sandbox() returns a proxy that records every call in the transcript.
    # no_events() is on the proxy in the inspect_ai releases this targets.
    no_events = getattr(sb, "no_events", None)
    if no_events is None:
        logger.warning(
            "this inspect_ai cannot hide sandbox events; the signed URL will be "
            "recorded in the eval log"
        )
        yield
        return
    with no_events():
        yield


@solver
def stage_problem_data(
    archives: dict[str, ProblemArchive],
    data_source: str,
    s3_uri: str | None = None,
    timeout: int = 7200,
    attempts: int = 3,
) -> Solver:
    """Download and extract the sample's problem data into the sandbox.

    Each attempt resolves a fresh URL, downloads the archive outside
    `/workspace`, checks its size and sha256 against the manifest, extracts it
    into `/workspace/data/` and deletes it. If every attempt fails, the solver
    raises, so the sample is marked ERRORED (and can be retried) rather than
    scored as a wrong answer.

    Args:
        archives: problem id -> manifest entry.
        data_source: "hf" or "s3".
        s3_uri: s3://bucket/prefix of the mirror (s3 only). Defaults to
            BMB_S3_URI.
        timeout: seconds allowed for one download and extraction.
        attempts: attempts before the sample errors.
    """
    if data_source not in ("hf", "s3"):
        raise ValueError(f"stage_problem_data needs 'hf' or 's3', got {data_source!r}")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        pid = str(state.metadata.get("problem_id") or state.sample_id)
        archive = archives[pid]
        sb = sandbox()
        mode = "bundle" if archive.bundled else "single"
        errors: list[str] = []
        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            try:
                url = await asyncio.to_thread(resolve_url, data_source, archive, s3_uri)
            except Exception as e:  # noqa: BLE001
                errors.append(f"attempt {attempt}: resolving the URL: {e}")
                continue
            await sb.exec(["mkdir", "-p", STAGE_DIR])
            with _without_events(sb):
                await sb.write_file(_URL_FILE, url)
            try:
                result = await sb.exec(
                    [
                        "sh",
                        "-c",
                        _STAGE_SCRIPT,
                        "stage",
                        pid,
                        archive.sha256,
                        str(archive.size),
                        mode,
                    ],
                    timeout=timeout,
                )
            except TimeoutError:
                errors.append(f"attempt {attempt}: timed out after {timeout}s")
                await sb.exec(["rm", "-rf", "/workspace/data", STAGE_DIR])
                continue
            if result.success:
                state.metadata["bmb_staging"] = {
                    "data_source": data_source,
                    "archive": archive.archive,
                    "archive_bytes": archive.size,
                    "data_bytes": int(result.stdout.split()[-1]),
                    "seconds": round(time.monotonic() - started, 1),
                    "attempts": attempt,
                }
                return state
            errors.append(f"attempt {attempt}: {result.stderr.strip()[-500:]}")
            await sb.exec(["rm", "-rf", "/workspace/data", STAGE_DIR])
        raise RuntimeError(
            f"BioMysteryBench problem {pid}: could not stage {archive.archive} from "
            f"{data_source}. " + " | ".join(errors)
        )

    return solve
