"""Acquire the frozen BENCH-016 DIV2K subset without storing the full ZIP.

This tool is intentionally acquisition-only: it streams the eight bound PNG members, hashes
their encoded bytes, and never decodes or resizes image content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import zlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterable, Mapping
from urllib.parse import urlparse

FROZEN_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
FROZEN_IDS = ("0057", "0171", "0285", "0399", "0513", "0627", "0741", "0785")
EXPECTED_MEMBERS = {image_id: f"DIV2K_train_HR/{image_id}.png" for image_id in FROZEN_IDS}
REMOTEZIP_VERSION = "0.12.3"
MANIFEST_NAME = "acquisition_manifest.json"
MANIFEST_HASH_NAME = "acquisition_manifest.sha256"
MANIFEST_SCHEMA = "structsplat.div2k-subset-acquisition.v1"
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_REQUEST_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class AcquisitionError(RuntimeError):
    """Raised when an acquisition binding or integrity condition fails."""


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _header(headers: Mapping[str, Any], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value).strip()
    return None


def bind_head(session: Any, url: str) -> dict[str, Any]:
    """Bind the remote object's identity using a body-free, redirect-following HEAD."""
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
        if content_length <= 0:
            raise AcquisitionError(f"non-positive HEAD Content-Length: {content_length}")
        return {
            "requested_url": url,
            "resolved_url": resolved_url,
            "status": int(response.status_code),
            "etag": _header(response.headers, "ETag"),
            "last_modified": _header(response.headers, "Last-Modified"),
            "content_length": content_length,
            "accept_ranges": _header(response.headers, "Accept-Ranges"),
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
    for field in ("resolved_url", "content_length", "etag", "last_modified"):
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
        response_url = str(response.url)
        if response_url != self.binding["resolved_url"]:
            raise AcquisitionError(
                f"range response URL changed: {response_url!r} != "
                f"{self.binding['resolved_url']!r}"
            )
        self.records.append(
            {
                "requested_range": requested,
                "content_range": raw_content_range,
                "content_length": end - start + 1,
                "response_url": response_url,
            }
        )
        return response


def validate_member_infos(infos: Iterable[Any]) -> list[Any]:
    """Resolve exactly one safe archive member for each frozen ID."""
    entries = list(infos)
    selected: dict[str, Any] = {}
    for info in entries:
        name = str(info.filename)
        stem = PurePosixPath(name).stem
        if stem not in EXPECTED_MEMBERS:
            continue
        expected = EXPECTED_MEMBERS[stem]
        if name != expected:
            raise AcquisitionError(f"unexpected archive member for frozen ID {stem}: {name!r}")
        if stem in selected:
            raise AcquisitionError(f"duplicate archive member: {name!r}")
        if info.is_dir():
            raise AcquisitionError(f"selected archive member is a directory: {name!r}")
        if int(info.flag_bits) & 1:
            raise AcquisitionError(f"selected archive member is encrypted: {name!r}")
        mode = (int(info.external_attr) >> 16) & 0xFFFF
        if stat.S_IFMT(mode) == stat.S_IFLNK:
            raise AcquisitionError(f"selected archive member is a symlink: {name!r}")
        if int(info.file_size) < 0 or int(info.compress_size) < 0:
            raise AcquisitionError(f"selected archive member has a negative size: {name!r}")
        if not 0 <= int(info.CRC) <= 0xFFFFFFFF:
            raise AcquisitionError(f"selected archive member has an invalid CRC: {name!r}")
        selected[stem] = info
    missing = [image_id for image_id in FROZEN_IDS if image_id not in selected]
    if missing:
        raise AcquisitionError(f"missing frozen archive members: {missing}")
    return [selected[image_id] for image_id in FROZEN_IDS]


def member_metadata(info: Any) -> dict[str, Any]:
    return {
        "archive_path": str(info.filename),
        "crc32": f"{int(info.CRC):08x}",
        "compressed_size": int(info.compress_size),
        "uncompressed_size": int(info.file_size),
        "compression_method": int(info.compress_type),
        "flag_bits": int(info.flag_bits),
    }


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
    if url != FROZEN_URL:
        raise AcquisitionError(f"URL is frozen to {FROZEN_URL}")
    destination = Path(outdir)
    _prepare_outdir(destination)

    if session is None:
        import requests

        session = requests.Session()
    session.headers.setdefault("Accept-Encoding", "identity")
    session.headers.setdefault("User-Agent", "StructSplat-BENCH-016-acquisition/1")
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
    try:
        infos = validate_member_infos(archive.infolist())
        for image_id, info in zip(FROZEN_IDS, infos):
            record = {"id": image_id, **member_metadata(info), "local_path": f"{image_id}.png"}
            with archive.open(info, "r") as source:
                record.update(stream_member(source, info, destination / record["local_path"]))
            records.append(record)
    finally:
        archive.close()
    if not audit.records:
        raise AcquisitionError("remotezip completed without any audited HTTP Range responses")

    after = bind_head(session, url)
    assert_same_remote_object(before, after)
    expected_before_manifest = {f"{image_id}.png" for image_id in FROZEN_IDS}
    observed_before_manifest = {path.name for path in destination.iterdir()}
    if observed_before_manifest != expected_before_manifest:
        raise AcquisitionError(
            f"unexpected output files before manifest: {sorted(observed_before_manifest)}"
        )
    manifest = write_manifest(
        destination,
        {
            "schema": MANIFEST_SCHEMA,
            "frozen_ids": list(FROZEN_IDS),
            "archive": before,
            "post_acquisition_head": after,
            "remotezip_version": remotezip_version,
            "members": records,
            "range_requests": audit.records,
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
    parser.add_argument("--url", default=FROZEN_URL, help="frozen official DIV2K training-HR URL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = acquire(args.outdir, url=args.url)
    print(f"acquired {len(manifest['members'])} frozen members")
    print(f"manifest payload sha256: {manifest['manifest_payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
