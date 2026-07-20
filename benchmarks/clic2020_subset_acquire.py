"""Acquire the frozen COMP-008 CLIC2020 Professional development subset.

The helper is deliberately acquisition-only.  It audits the complete ZIP central directory,
streams only the eight bound development PNG members, binds confirmation metadata without
extracting its payloads, and never imports an image decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
import unicodedata
import zlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterable, Mapping
from urllib.parse import urlparse

FROZEN_URL = "https://storage.googleapis.com/clic_datasets/clic2020_professional_valid.zip"
EXPECTED_CONTENT_LENGTH = 134_862_753
EXPECTED_GCS_GENERATION = "1755735394217541"
DEVELOPMENT_NAMES = (
    "nomao-saeki-33553",
    "martyn-seddon-220",
    "zugr-108",
    "jason-briscoe-149782",
    "martin-wessely-211",
    "stefan-kunze-26931",
    "vita-vilcina-3055",
    "philippe-wuyts-45997",
)
CONFIRMATION_NAMES = (
    "lobostudio-hamburg-75377",
    "gian-reto-tarnutzer-45212",
    "roberto-nickson-48063",
    "wojciech-szaturski-3611",
    "philipp-reiner-207",
    "sergey-zolkin-21232",
    "todd-quackenbush-222",
    "alexander-shustov-73",
)
FROZEN_NAMES = DEVELOPMENT_NAMES + CONFIRMATION_NAMES
REMOTEZIP_VERSION = "0.12.3"
MANIFEST_NAME = "acquisition_manifest.json"
MANIFEST_HASH_NAME = "acquisition_manifest.sha256"
MANIFEST_SCHEMA = "structsplat.clic2020-professional-subset-acquisition.v1"
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_REQUEST_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


class AcquisitionError(RuntimeError):
    """Raised when a frozen acquisition or integrity condition fails."""


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _process_stage_record(started_utc: str, started_perf: float) -> dict[str, Any]:
    """Capture the actual acquisition invocation without weakening the data seal."""
    return {
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started_perf,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "working_directory": str(Path.cwd().resolve()),
        "python_executable": str(Path(sys.executable).resolve()),
        "invocation_argv": [str(value) for value in sys.argv],
    }


def _header(headers: Mapping[str, Any], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value).strip()
    return None


def bind_head(session: Any, url: str) -> dict[str, Any]:
    """Bind the remote object with a body-free, redirect-following HEAD request."""
    response = session.head(
        url,
        allow_redirects=True,
        timeout=(10, 60),
        headers={"Accept-Encoding": "identity"},
    )
    try:
        response.raise_for_status()
        if int(response.status_code) != 200:
            raise AcquisitionError(f"HEAD returned status {response.status_code}, expected 200")
        resolved_url = str(response.url)
        if urlparse(resolved_url).scheme != "https":
            raise AcquisitionError(f"refusing non-HTTPS resolved URL: {resolved_url}")
        raw_length = _header(response.headers, "Content-Length")
        try:
            content_length = int(raw_length or "")
        except ValueError as exc:
            raise AcquisitionError(f"invalid HEAD Content-Length: {raw_length!r}") from exc
        if content_length != EXPECTED_CONTENT_LENGTH:
            raise AcquisitionError(
                f"frozen Content-Length mismatch: {content_length} != {EXPECTED_CONTENT_LENGTH}"
            )
        generation = _header(response.headers, "X-Goog-Generation")
        if generation != EXPECTED_GCS_GENERATION:
            raise AcquisitionError(
                f"frozen GCS generation mismatch: {generation!r} != {EXPECTED_GCS_GENERATION!r}"
            )
        return {
            "requested_url": url,
            "resolved_url": resolved_url,
            "status": int(response.status_code),
            "etag": _header(response.headers, "ETag"),
            "last_modified": _header(response.headers, "Last-Modified"),
            "content_length": content_length,
            "accept_ranges": _header(response.headers, "Accept-Ranges"),
            "gcs_generation": generation,
        }
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            close()


def if_range_headers(binding: Mapping[str, Any]) -> dict[str, str]:
    etag = binding.get("etag")
    if isinstance(etag, str) and etag and not etag.lower().startswith("w/"):
        return {"If-Range": etag, "Accept-Encoding": "identity"}
    modified = binding.get("last_modified")
    if isinstance(modified, str) and modified:
        return {"If-Range": modified, "Accept-Encoding": "identity"}
    raise AcquisitionError("HEAD supplied neither a strong ETag nor Last-Modified for If-Range")


def assert_same_remote_object(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    for field in ("resolved_url", "content_length", "etag", "last_modified", "gcs_generation"):
        if before.get(field) != after.get(field):
            raise AcquisitionError(
                f"remote object changed during acquisition: {field} "
                f"{before.get(field)!r} -> {after.get(field)!r}"
            )


def _expected_response_range(requested: str, total: int) -> tuple[int, int]:
    match = _REQUEST_RANGE.fullmatch(requested)
    if match is None:
        raise AcquisitionError(f"invalid requested Range header: {requested!r}")
    first, last = match.groups()
    if not first:
        if not last or int(last) <= 0:
            raise AcquisitionError(f"invalid suffix Range header: {requested!r}")
        count = int(last)
        return max(0, total - count), total - 1
    start = int(first)
    if start >= total:
        raise AcquisitionError(f"requested range starts beyond object: {requested!r}")
    if not last:
        return start, total - 1
    end = int(last)
    if end < start:
        raise AcquisitionError(f"descending Range header: {requested!r}")
    return start, min(end, total - 1)


class RangeAudit:
    """Requests response hook that rejects non-exact HTTP Range behavior."""

    def __init__(self, binding: Mapping[str, Any]) -> None:
        self.binding = dict(binding)
        self.records: list[dict[str, Any]] = []

    def __call__(self, response: Any, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        request = response.request
        if str(request.method).upper() != "GET":
            return response
        requested = _header(request.headers, "Range")
        if requested is None:
            raise AcquisitionError("archive GET omitted the Range header")
        if int(response.status_code) != 206:
            raise AcquisitionError(
                f"archive GET returned {response.status_code}, expected HTTP 206"
            )
        raw_content_range = _header(response.headers, "Content-Range")
        match = _CONTENT_RANGE.fullmatch(raw_content_range or "")
        if match is None:
            raise AcquisitionError(f"invalid Content-Range: {raw_content_range!r}")
        start, end, total = (int(value) for value in match.groups())
        expected_total = int(self.binding["content_length"])
        if total != expected_total:
            raise AcquisitionError(f"Content-Range total changed: {total} != {expected_total}")
        if (start, end) != _expected_response_range(requested, total):
            raise AcquisitionError(
                f"server returned bytes {start}-{end} for requested {requested!r}"
            )
        response_length = _header(response.headers, "Content-Length")
        if response_length is not None and int(response_length) != end - start + 1:
            raise AcquisitionError("range response Content-Length does not match Content-Range")
        for header_name, field in (("ETag", "etag"), ("Last-Modified", "last_modified")):
            expected = self.binding.get(field)
            observed = _header(response.headers, header_name)
            if expected is not None and observed is not None and expected != observed:
                raise AcquisitionError(f"range response {header_name} changed")
        observed_generation = _header(response.headers, "X-Goog-Generation")
        if observed_generation != EXPECTED_GCS_GENERATION:
            raise AcquisitionError("range response GCS generation changed")
        bound_generation = self.binding.get("gcs_generation")
        if observed_generation != bound_generation:
            raise AcquisitionError("range response GCS generation differs from HEAD")
        response_url = str(response.url)
        if response_url != self.binding["resolved_url"]:
            raise AcquisitionError(
                f"range response URL changed: {response_url!r} != {self.binding['resolved_url']!r}"
            )
        self.records.append(
            {
                "requested_range": requested,
                "content_range": raw_content_range,
                "content_length": end - start + 1,
                "response_url": response_url,
                "gcs_generation": observed_generation,
            }
        )
        return response


def _validated_archive_path(info: Any) -> str:
    raw = str(info.filename)
    if not raw or "\x00" in raw:
        raise AcquisitionError(f"invalid empty or NUL archive path: {raw!r}")
    if "\\" in raw:
        raise AcquisitionError(f"archive path contains a backslash: {raw!r}")
    normalized = unicodedata.normalize("NFC", raw)
    if normalized.startswith("/") or _WINDOWS_DRIVE.match(normalized):
        raise AcquisitionError(f"absolute archive path is forbidden: {raw!r}")
    is_directory = bool(info.is_dir())
    core = normalized[:-1] if is_directory and normalized.endswith("/") else normalized
    parts = core.split("/")
    if not core or any(part in {"", ".", ".."} for part in parts):
        raise AcquisitionError(f"unsafe traversal or empty archive path component: {raw!r}")
    if not is_directory and normalized.endswith("/"):
        raise AcquisitionError(f"non-directory archive path ends with '/': {raw!r}")
    return normalized


def member_metadata(info: Any, *, normalized_path: str | None = None) -> dict[str, Any]:
    normalized = normalized_path or unicodedata.normalize("NFC", str(info.filename))
    mode = (int(info.external_attr) >> 16) & 0xFFFF
    return {
        "archive_path": str(info.filename),
        "archive_path_nfc": normalized,
        "crc32": f"{int(info.CRC):08x}",
        "compressed_size": int(info.compress_size),
        "uncompressed_size": int(info.file_size),
        "compression_method": int(info.compress_type),
        "flag_bits": int(info.flag_bits),
        "unix_mode": mode,
        "is_directory": bool(info.is_dir()),
    }


def validate_member_infos(infos: Iterable[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    """Audit the complete central directory, then resolve the sixteen bound PNGs."""
    entries = list(infos)
    selected: dict[str, Any] = {}
    seen_paths: set[str] = set()
    seen_png_stems: set[str] = set()
    archive_index: list[dict[str, Any]] = []
    for info in entries:
        normalized = _validated_archive_path(info)
        if normalized in seen_paths:
            raise AcquisitionError(f"duplicate normalized archive path: {normalized!r}")
        seen_paths.add(normalized)
        if int(info.flag_bits) & 1:
            raise AcquisitionError(f"encrypted archive member is forbidden: {normalized!r}")
        mode = (int(info.external_attr) >> 16) & 0xFFFF
        if stat.S_IFMT(mode) == stat.S_IFLNK:
            raise AcquisitionError(f"symlink archive member is forbidden: {normalized!r}")
        if int(info.file_size) < 0 or int(info.compress_size) < 0:
            raise AcquisitionError(f"archive member has a negative size: {normalized!r}")
        if not 0 <= int(info.CRC) <= 0xFFFFFFFF:
            raise AcquisitionError(f"archive member has an invalid CRC: {normalized!r}")
        archive_index.append(member_metadata(info, normalized_path=normalized))
        if info.is_dir():
            continue
        path = PurePosixPath(normalized)
        stem = path.stem
        suffix = path.suffix
        if stem in FROZEN_NAMES and suffix != ".png":
            raise AcquisitionError(
                f"bound basename must use the exact lowercase .png suffix: {normalized!r}"
            )
        if suffix != ".png":
            continue
        if stem in seen_png_stems:
            raise AcquisitionError(f"duplicate exact PNG stem in archive: {stem!r}")
        seen_png_stems.add(stem)
        if stem in FROZEN_NAMES:
            selected[stem] = info
    missing = [name for name in FROZEN_NAMES if name not in selected]
    if missing:
        raise AcquisitionError(f"missing frozen archive members: {missing}")
    return [selected[name] for name in FROZEN_NAMES], archive_index


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def stream_member(source: BinaryIO, info: Any, destination: Path) -> dict[str, Any]:
    """Hash and atomically persist one encoded member, checking size and CRC."""
    temporary = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    crc = 0
    size = 0
    try:
        with temporary.open("xb") as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                crc = zlib.crc32(chunk, crc)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size != int(info.file_size):
            raise AcquisitionError(
                f"member size mismatch for {info.filename}: {size} != {info.file_size}"
            )
        crc &= 0xFFFFFFFF
        if crc != int(info.CRC):
            raise AcquisitionError(
                f"member CRC mismatch for {info.filename}: {crc:08x} != {int(info.CRC):08x}"
            )
        if destination.exists():
            raise AcquisitionError(f"destination appeared during acquisition: {destination}")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"byte_size": size, "sha256": digest.hexdigest(), "download_crc32": f"{crc:08x}"}


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if path.exists():
            raise AcquisitionError(f"destination appeared during acquisition: {path}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_manifest(outdir: Path, core: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(core)
    manifest["manifest_payload_sha256"] = _sha256(canonical_json(manifest))
    payload = canonical_json(manifest)
    file_hash = _sha256(payload)
    _atomic_write(outdir / MANIFEST_NAME, payload)
    _atomic_write(
        outdir / MANIFEST_HASH_NAME,
        f"{file_hash}  {MANIFEST_NAME}\n".encode("ascii"),
    )
    return manifest


def _prepare_outdir(outdir: Path) -> None:
    if outdir.exists() and not outdir.is_dir():
        raise AcquisitionError(f"output path is not a directory: {outdir}")
    if outdir.exists() and any(outdir.iterdir()):
        raise AcquisitionError(f"output directory must be empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)


def _load_remotezip() -> tuple[Callable[..., Any], str]:
    try:
        installed = version("remotezip")
    except PackageNotFoundError as exc:
        raise AcquisitionError("remotezip==0.12.3 is required") from exc
    if installed != REMOTEZIP_VERSION:
        raise AcquisitionError(
            f"remotezip version drift: expected {REMOTEZIP_VERSION}, found {installed}"
        )
    from remotezip import RemoteZip

    return RemoteZip, installed


def acquire(
    outdir: str | Path,
    *,
    url: str = FROZEN_URL,
    session: Any | None = None,
    remote_zip_factory: Callable[..., Any] | None = None,
    remotezip_version: str | None = None,
) -> dict[str, Any]:
    started_utc = _utc_now()
    started_perf = time.perf_counter()
    if url != FROZEN_URL:
        raise AcquisitionError(f"URL is frozen to {FROZEN_URL}")
    destination = Path(outdir)
    _prepare_outdir(destination)

    if session is None:
        import requests

        session = requests.Session()
    session.headers.setdefault("Accept-Encoding", "identity")
    session.headers.setdefault("User-Agent", "StructSplat-COMP-008-acquisition/1")
    if remote_zip_factory is None:
        remote_zip_factory, remotezip_version = _load_remotezip()
    elif remotezip_version is None:
        remotezip_version = "injected-test-double"

    before = bind_head(session, url)
    audit = RangeAudit(before)
    session.hooks.setdefault("response", []).append(audit)
    archive = remote_zip_factory(
        before["resolved_url"],
        session=session,
        initial_buffer_size=65536,
        support_suffix_range=True,
        headers=if_range_headers(before),
    )
    records: list[dict[str, Any]] = []
    archive_index: list[dict[str, Any]]
    try:
        infos, archive_index = validate_member_infos(archive.infolist())
        for name, info in zip(FROZEN_NAMES, infos):
            partition = "development" if name in DEVELOPMENT_NAMES else "confirmation"
            record = {
                "name": name,
                "partition": partition,
                **member_metadata(info),
                "payload_extracted": partition == "development",
                "local_path": f"{name}.png" if partition == "development" else None,
            }
            if partition == "development":
                with archive.open(info, "r") as source:
                    record.update(
                        stream_member(source, info, destination / str(record["local_path"]))
                    )
            records.append(record)
    finally:
        archive.close()
    if not audit.records:
        raise AcquisitionError("remotezip completed without any audited HTTP Range responses")

    after = bind_head(session, url)
    assert_same_remote_object(before, after)
    expected_before_manifest = {f"{name}.png" for name in DEVELOPMENT_NAMES}
    observed_before_manifest = {path.name for path in destination.iterdir()}
    if observed_before_manifest != expected_before_manifest:
        raise AcquisitionError(
            f"unexpected output files before manifest: {sorted(observed_before_manifest)}"
        )
    manifest = write_manifest(
        destination,
        {
            "schema": MANIFEST_SCHEMA,
            "development_names": list(DEVELOPMENT_NAMES),
            "confirmation_names": list(CONFIRMATION_NAMES),
            "archive": before,
            "post_acquisition_head": after,
            "remotezip_version": remotezip_version,
            "archive_index": archive_index,
            "archive_index_payload_sha256": _sha256(canonical_json(archive_index)),
            "members": records,
            "range_requests": audit.records,
            "stage_process": _process_stage_record(started_utc, started_perf),
        },
    )
    expected_final = expected_before_manifest | {MANIFEST_NAME, MANIFEST_HASH_NAME}
    observed_final = {path.name for path in destination.iterdir()}
    if observed_final != expected_final:
        raise AcquisitionError(f"unexpected final output files: {sorted(observed_final)}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--url", default=FROZEN_URL, help="frozen official CLIC2020 URL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = acquire(args.outdir, url=args.url)
    extracted = sum(bool(record["payload_extracted"]) for record in manifest["members"])
    print(f"acquired {extracted} frozen development PNG members")
    print(f"manifest payload sha256: {manifest['manifest_payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
