import hashlib
import io
import json
import zlib
from pathlib import PurePosixPath
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipInfo

import pytest

from benchmarks import div2k_subset_acquire as A


def _info(image_id: str, payload: bytes = b"png-bytes") -> ZipInfo:
    info = ZipInfo(A.EXPECTED_MEMBERS[image_id])
    info.file_size = len(payload)
    info.compress_size = max(1, len(payload) - 1)
    info.CRC = zlib.crc32(payload) & 0xFFFFFFFF
    info.compress_type = ZIP_DEFLATED
    info.flag_bits = 0
    info.external_attr = 0o100644 << 16
    return info


def _infos(payloads=None):
    payloads = payloads or {image_id: b"png-" + image_id.encode() for image_id in A.FROZEN_IDS}
    return [_info(image_id, payloads[image_id]) for image_id in A.FROZEN_IDS]


class _Response:
    def __init__(self, *, status, url=A.FROZEN_URL, headers=None, request=None):
        self.status_code = status
        self.url = url
        self.headers = headers or {}
        self.request = request
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        self.closed = True


def _head_response(**overrides):
    headers = {
        "Content-Length": "1000",
        "ETag": '"frozen"',
        "Last-Modified": "Wed, 15 Jul 2026 12:00:00 GMT",
        "Accept-Ranges": "bytes",
    }
    headers.update(overrides.pop("headers", {}))
    return _Response(status=200, headers=headers, **overrides)


class _Session:
    def __init__(self, heads):
        self._heads = iter(heads)
        self.headers = {}
        self.hooks = {}

    def head(self, url, **kwargs):
        del url, kwargs
        return next(self._heads)


def _range_response(
    requested="bytes=-100",
    *,
    status=206,
    content_range="bytes 900-999/1000",
    response_url=A.FROZEN_URL,
    extra_headers=None,
):
    headers = {
        "Content-Range": content_range,
        "Content-Length": "100",
        "ETag": '"frozen"',
        "Last-Modified": "Wed, 15 Jul 2026 12:00:00 GMT",
    }
    headers.update(extra_headers or {})
    request = SimpleNamespace(method="GET", headers={"Range": requested})
    return _Response(status=status, url=response_url, headers=headers, request=request)


def test_frozen_ids_and_members_are_exact():
    assert A.FROZEN_IDS == ("0057", "0171", "0285", "0399", "0513", "0627", "0741", "0785")
    assert list(A.EXPECTED_MEMBERS.values()) == [
        f"DIV2K_train_HR/{image_id}.png" for image_id in A.FROZEN_IDS
    ]


def test_head_binding_records_required_headers_and_closes_response():
    response = _head_response()
    binding = A.bind_head(_Session([response]), A.FROZEN_URL)
    assert response.closed
    assert binding == {
        "requested_url": A.FROZEN_URL,
        "resolved_url": A.FROZEN_URL,
        "status": 200,
        "etag": '"frozen"',
        "last_modified": "Wed, 15 Jul 2026 12:00:00 GMT",
        "content_length": 1000,
        "accept_ranges": "bytes",
    }
    assert A.if_range_headers(binding)["If-Range"] == '"frozen"'


def test_head_binding_rejects_unbound_or_downgraded_objects():
    with pytest.raises(A.AcquisitionError, match="Content-Length"):
        A.bind_head(_Session([_head_response(headers={"Content-Length": ""})]), A.FROZEN_URL)
    with pytest.raises(A.AcquisitionError, match="non-HTTPS"):
        A.bind_head(
            _Session([_head_response(url="http://example.test/archive.zip")]), A.FROZEN_URL
        )
    binding = A.bind_head(_Session([_head_response()]), A.FROZEN_URL)
    binding["etag"] = "W/weak"
    assert A.if_range_headers(binding)["If-Range"] == binding["last_modified"]
    binding["last_modified"] = None
    with pytest.raises(A.AcquisitionError, match="neither"):
        A.if_range_headers(binding)


def test_post_head_must_match_every_identity_field():
    before = A.bind_head(_Session([_head_response()]), A.FROZEN_URL)
    A.assert_same_remote_object(before, dict(before))
    for field, changed in (
        ("resolved_url", "https://example.test/changed.zip"),
        ("content_length", 999),
        ("etag", '"changed"'),
        ("last_modified", "Thu, 16 Jul 2026 12:00:00 GMT"),
    ):
        after = dict(before)
        after[field] = changed
        with pytest.raises(A.AcquisitionError, match=field):
            A.assert_same_remote_object(before, after)


def test_range_audit_accepts_exact_suffix_and_explicit_ranges():
    binding = A.bind_head(_Session([_head_response()]), A.FROZEN_URL)
    audit = A.RangeAudit(binding)
    suffix = _range_response()
    assert audit(suffix) is suffix
    explicit = _range_response(
        "bytes=100-149",
        content_range="bytes 100-149/1000",
        extra_headers={"Content-Length": "50"},
    )
    audit(explicit)
    assert [row["requested_range"] for row in audit.records] == [
        "bytes=-100",
        "bytes=100-149",
    ]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_range_response(status=200), "expected HTTP 206"),
        (_range_response(content_range="bytes 900-999/999"), "total changed"),
        (_range_response(content_range="bytes 899-998/1000"), "server returned bytes"),
        (_range_response(extra_headers={"Content-Length": "99"}), "does not match"),
        (_range_response(extra_headers={"ETag": '"changed"'}), "ETag changed"),
        (
            _range_response(response_url="https://example.test/other.zip"),
            "response URL changed",
        ),
    ],
)
def test_range_audit_rejects_nonexact_responses(response, message):
    binding = A.bind_head(_Session([_head_response()]), A.FROZEN_URL)
    with pytest.raises(A.AcquisitionError, match=message):
        A.RangeAudit(binding)(response)


def test_member_validation_accepts_only_exact_safe_unique_entries():
    infos = _infos()
    assert A.validate_member_infos(infos) == infos

    missing = infos[:-1]
    with pytest.raises(A.AcquisitionError, match="missing"):
        A.validate_member_infos(missing)

    duplicate = infos + [_info(A.FROZEN_IDS[0], b"different")]
    with pytest.raises(A.AcquisitionError, match="duplicate"):
        A.validate_member_infos(duplicate)

    alternate = _info(A.FROZEN_IDS[0])
    alternate.filename = f"other/{A.FROZEN_IDS[0]}.png"
    with pytest.raises(A.AcquisitionError, match="unexpected archive member"):
        A.validate_member_infos(infos[1:] + [alternate])


@pytest.mark.parametrize("mutation", ["directory", "encrypted", "symlink", "negative_size"])
def test_member_validation_rejects_unsafe_entries(mutation):
    infos = _infos()
    info = infos[0]
    if mutation == "directory":
        info.filename += "/"
    elif mutation == "encrypted":
        info.flag_bits |= 1
    elif mutation == "symlink":
        info.external_attr = 0o120777 << 16
    else:
        info.file_size = -1
    with pytest.raises(A.AcquisitionError):
        A.validate_member_infos(infos)


def test_stream_member_hashes_and_atomically_checks_size_and_crc(tmp_path):
    payload = b"encoded-png-payload"
    info = _info("0057", payload)
    destination = tmp_path / "0057.png"
    result = A.stream_member(io.BytesIO(payload), info, destination)
    assert destination.read_bytes() == payload
    assert result == {
        "byte_size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "download_crc32": f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
    }
    assert not (tmp_path / "0057.png.part").exists()

    bad_info = _info("0171", payload)
    bad_info.CRC ^= 1
    with pytest.raises(A.AcquisitionError, match="CRC mismatch"):
        A.stream_member(io.BytesIO(payload), bad_info, tmp_path / "0171.png")
    assert not (tmp_path / "0171.png").exists()
    assert not (tmp_path / "0171.png.part").exists()


class _Archive:
    def __init__(self, infos, payloads):
        self.infos = infos
        self.payloads = payloads
        self.closed = False

    def infolist(self):
        return self.infos

    def open(self, info, mode):
        assert mode == "r"
        return io.BytesIO(self.payloads[PurePosixPath(info.filename).stem])

    def close(self):
        self.closed = True


def test_complete_acquisition_with_fakes_writes_self_hashed_manifest(tmp_path):
    payloads = {image_id: b"encoded-" + image_id.encode() for image_id in A.FROZEN_IDS}
    infos = _infos(payloads)
    session = _Session([_head_response(), _head_response()])
    archive = _Archive(infos, payloads)

    def factory(url, *, session, **kwargs):
        assert url == A.FROZEN_URL
        assert kwargs == {
            "initial_buffer_size": 65536,
            "support_suffix_range": True,
            "headers": {
                "If-Range": '"frozen"',
                "Accept-Encoding": "identity",
            },
        }
        assert session.headers["Accept-Encoding"] == "identity"
        for hook in session.hooks["response"]:
            hook(_range_response())
        return archive

    outdir = tmp_path / "source"
    manifest = A.acquire(
        outdir,
        session=session,
        remote_zip_factory=factory,
        remotezip_version=A.REMOTEZIP_VERSION,
    )
    assert archive.closed
    assert {path.name for path in outdir.iterdir()} == {
        *(f"{image_id}.png" for image_id in A.FROZEN_IDS),
        A.MANIFEST_NAME,
        A.MANIFEST_HASH_NAME,
    }
    for image_id, payload in payloads.items():
        assert (outdir / f"{image_id}.png").read_bytes() == payload

    stored = json.loads((outdir / A.MANIFEST_NAME).read_text())
    unhashed = {key: value for key, value in stored.items() if key != "manifest_payload_sha256"}
    assert stored == manifest
    assert stored["manifest_payload_sha256"] == hashlib.sha256(
        A.canonical_json(unhashed)
    ).hexdigest()
    file_hash = hashlib.sha256((outdir / A.MANIFEST_NAME).read_bytes()).hexdigest()
    assert (outdir / A.MANIFEST_HASH_NAME).read_text() == (
        f"{file_hash}  {A.MANIFEST_NAME}\n"
    )


def test_acquisition_refuses_nonempty_destination_before_head(tmp_path):
    outdir = tmp_path / "source"
    outdir.mkdir()
    (outdir / "foreign").write_text("do not overwrite")
    session = _Session([])
    with pytest.raises(A.AcquisitionError, match="must be empty"):
        A.acquire(outdir, session=session, remote_zip_factory=lambda *args, **kwargs: None)


def test_cli_has_no_id_override_and_rejects_url_drift(tmp_path):
    parser = A.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--outdir", str(tmp_path), "--ids", "0057"])
    with pytest.raises(A.AcquisitionError, match="URL is frozen"):
        A.acquire(
            tmp_path / "source",
            url="https://example.test/not-div2k.zip",
            session=_Session([]),
            remote_zip_factory=lambda *args, **kwargs: None,
        )
