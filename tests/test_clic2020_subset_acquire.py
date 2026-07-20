import ast
import hashlib
import io
import json
import unicodedata
import zlib
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipInfo

import pytest

from benchmarks import clic2020_subset_acquire as A


def _info(
    name: str, payload: bytes = b"encoded-png", *, archive_path: str | None = None
) -> ZipInfo:
    info = ZipInfo(archive_path or f"professional_valid/{name}.png")
    info.file_size = len(payload)
    info.compress_size = max(1, len(payload) - 1)
    info.CRC = zlib.crc32(payload) & 0xFFFFFFFF
    info.compress_type = ZIP_DEFLATED
    info.flag_bits = 0
    info.external_attr = 0o100644 << 16
    return info


def _infos(payloads=None):
    payloads = payloads or {name: b"png-" + name.encode() for name in A.FROZEN_NAMES}
    return [_info(name, payloads[name]) for name in A.FROZEN_NAMES]


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
        "Content-Length": str(A.EXPECTED_CONTENT_LENGTH),
        "ETag": '"frozen"',
        "Last-Modified": "Thu, 21 Aug 2025 00:16:34 GMT",
        "Accept-Ranges": "bytes",
        "X-Goog-Generation": A.EXPECTED_GCS_GENERATION,
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
    content_range=None,
    response_url=A.FROZEN_URL,
    extra_headers=None,
):
    total = A.EXPECTED_CONTENT_LENGTH
    content_range = content_range or f"bytes {total - 100}-{total - 1}/{total}"
    headers = {
        "Content-Range": content_range,
        "Content-Length": "100",
        "ETag": '"frozen"',
        "Last-Modified": "Thu, 21 Aug 2025 00:16:34 GMT",
        "X-Goog-Generation": A.EXPECTED_GCS_GENERATION,
    }
    headers.update(extra_headers or {})
    request = SimpleNamespace(method="GET", headers={"Range": requested})
    return _Response(status=status, url=response_url, headers=headers, request=request)


def test_frozen_url_object_identity_and_partitions_are_exact():
    assert A.FROZEN_URL == (
        "https://storage.googleapis.com/clic_datasets/clic2020_professional_valid.zip"
    )
    assert A.EXPECTED_CONTENT_LENGTH == 134_862_753
    assert A.EXPECTED_GCS_GENERATION == "1755735394217541"
    assert A.REMOTEZIP_VERSION == "0.12.3"
    assert A.DEVELOPMENT_NAMES == (
        "nomao-saeki-33553",
        "martyn-seddon-220",
        "zugr-108",
        "jason-briscoe-149782",
        "martin-wessely-211",
        "stefan-kunze-26931",
        "vita-vilcina-3055",
        "philippe-wuyts-45997",
    )
    assert A.CONFIRMATION_NAMES == (
        "lobostudio-hamburg-75377",
        "gian-reto-tarnutzer-45212",
        "roberto-nickson-48063",
        "wojciech-szaturski-3611",
        "philipp-reiner-207",
        "sergey-zolkin-21232",
        "todd-quackenbush-222",
        "alexander-shustov-73",
    )
    assert A.FROZEN_NAMES == A.DEVELOPMENT_NAMES + A.CONFIRMATION_NAMES


def test_head_binding_enforces_length_and_exposed_gcs_generation():
    response = _head_response()
    binding = A.bind_head(_Session([response]), A.FROZEN_URL)
    assert response.closed
    assert binding == {
        "requested_url": A.FROZEN_URL,
        "resolved_url": A.FROZEN_URL,
        "status": 200,
        "etag": '"frozen"',
        "last_modified": "Thu, 21 Aug 2025 00:16:34 GMT",
        "content_length": A.EXPECTED_CONTENT_LENGTH,
        "accept_ranges": "bytes",
        "gcs_generation": A.EXPECTED_GCS_GENERATION,
    }
    assert A.if_range_headers(binding)["If-Range"] == '"frozen"'

    wrong_length = _head_response(headers={"Content-Length": "1"})
    with pytest.raises(A.AcquisitionError, match="Content-Length mismatch"):
        A.bind_head(_Session([wrong_length]), A.FROZEN_URL)
    wrong_generation = _head_response(headers={"X-Goog-Generation": "2"})
    with pytest.raises(A.AcquisitionError, match="generation mismatch"):
        A.bind_head(_Session([wrong_generation]), A.FROZEN_URL)


def test_head_requires_generation_and_rejects_downgrade_and_weak_identity():
    response = _head_response()
    response.headers.pop("X-Goog-Generation")
    with pytest.raises(A.AcquisitionError, match="generation mismatch"):
        A.bind_head(_Session([response]), A.FROZEN_URL)

    with pytest.raises(A.AcquisitionError, match="non-HTTPS"):
        A.bind_head(_Session([_head_response(url="http://example.test/archive.zip")]), A.FROZEN_URL)
    binding = A.bind_head(_Session([_head_response()]), A.FROZEN_URL)
    binding["etag"] = "W/weak"
    assert A.if_range_headers(binding)["If-Range"] == binding["last_modified"]
    binding["last_modified"] = None
    with pytest.raises(A.AcquisitionError, match="neither"):
        A.if_range_headers(binding)


def test_post_head_must_match_all_identity_fields_including_generation():
    before = A.bind_head(_Session([_head_response()]), A.FROZEN_URL)
    A.assert_same_remote_object(before, dict(before))
    for field, changed in (
        ("resolved_url", "https://example.test/changed.zip"),
        ("content_length", A.EXPECTED_CONTENT_LENGTH - 1),
        ("etag", '"changed"'),
        ("last_modified", "Fri, 22 Aug 2025 00:16:34 GMT"),
        ("gcs_generation", "1"),
    ):
        after = dict(before)
        after[field] = changed
        with pytest.raises(A.AcquisitionError, match=field):
            A.assert_same_remote_object(before, after)


def test_range_audit_accepts_exact_ranges_and_rejects_generation_drift():
    total = A.EXPECTED_CONTENT_LENGTH
    binding = A.bind_head(_Session([_head_response()]), A.FROZEN_URL)
    audit = A.RangeAudit(binding)
    suffix = _range_response()
    assert audit(suffix) is suffix
    explicit = _range_response(
        "bytes=100-149",
        content_range=f"bytes 100-149/{total}",
        extra_headers={"Content-Length": "50"},
    )
    audit(explicit)
    assert [row["requested_range"] for row in audit.records] == [
        "bytes=-100",
        "bytes=100-149",
    ]
    with pytest.raises(A.AcquisitionError, match="generation changed"):
        audit(_range_response(extra_headers={"X-Goog-Generation": "1"}))
    missing_generation = _range_response()
    missing_generation.headers.pop("X-Goog-Generation")
    with pytest.raises(A.AcquisitionError, match="generation changed"):
        audit(missing_generation)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_range_response(status=200), "expected HTTP 206"),
        (_range_response(content_range="bytes 0-99/100"), "total changed"),
        (
            _range_response(content_range=f"bytes 0-99/{A.EXPECTED_CONTENT_LENGTH}"),
            "server returned",
        ),
        (_range_response(extra_headers={"Content-Length": "99"}), "does not match"),
        (_range_response(extra_headers={"ETag": '"changed"'}), "ETag changed"),
        (_range_response(response_url="https://example.test/other.zip"), "response URL changed"),
    ],
)
def test_range_audit_rejects_nonexact_responses(response, message):
    binding = A.bind_head(_Session([_head_response()]), A.FROZEN_URL)
    with pytest.raises(A.AcquisitionError, match=message):
        A.RangeAudit(binding)(response)


def test_complete_central_directory_validation_selects_exact_names_in_frozen_order():
    infos = _infos()
    selected, index = A.validate_member_infos(infos)
    assert selected == infos
    assert [PurePosixPath(row["archive_path_nfc"]).stem for row in index] == list(A.FROZEN_NAMES)

    with pytest.raises(A.AcquisitionError, match="missing frozen"):
        A.validate_member_infos(infos[:-1])
    duplicate = infos + [_info(A.FROZEN_NAMES[0], b"different")]
    with pytest.raises(A.AcquisitionError, match="duplicate normalized archive path"):
        A.validate_member_infos(duplicate)
    wrong_suffix = _infos()
    wrong_suffix[0].filename = wrong_suffix[0].filename[:-4] + ".PNG"
    with pytest.raises(A.AcquisitionError, match="lowercase .png"):
        A.validate_member_infos(wrong_suffix)


@pytest.mark.parametrize(
    ("archive_path", "mutation", "message"),
    [
        ("../escape.txt", None, "traversal"),
        ("/absolute.txt", None, "absolute"),
        ("C:/absolute.txt", None, "absolute"),
        ("folder\\escape.txt", None, "backslash"),
        ("safe/encrypted.txt", "encrypted", "encrypted"),
        ("safe/link.txt", "symlink", "symlink"),
    ],
)
def test_validation_rejects_unsafe_entries_anywhere_in_archive(archive_path, mutation, message):
    foreign = _info("foreign", archive_path=archive_path)
    if mutation == "encrypted":
        foreign.flag_bits |= 1
    elif mutation == "symlink":
        foreign.external_attr = 0o120777 << 16
    with pytest.raises(A.AcquisitionError, match=message):
        A.validate_member_infos(_infos() + [foreign])


def test_validation_normalizes_nfc_and_rejects_normalized_or_png_stem_duplicates():
    decomposed = _info("foreign", archive_path="other/cafe\u0301.txt")
    _, index = A.validate_member_infos(_infos() + [decomposed])
    assert index[-1]["archive_path_nfc"] == unicodedata.normalize("NFC", decomposed.filename)

    composed = _info("foreign2", archive_path="other/caf\u00e9.txt")
    with pytest.raises(A.AcquisitionError, match="duplicate normalized archive path"):
        A.validate_member_infos(_infos() + [decomposed, composed])

    duplicate_stem_a = _info("foreign", archive_path="a/foreign.png")
    duplicate_stem_b = _info("foreign", archive_path="b/foreign.png")
    with pytest.raises(A.AcquisitionError, match="duplicate exact PNG stem"):
        A.validate_member_infos(_infos() + [duplicate_stem_a, duplicate_stem_b])

    # Exact case-sensitive uniqueness keeps distinct case variants distinct.
    A.validate_member_infos(
        _infos()
        + [
            _info("foreign", archive_path="a/foreign.png"),
            _info("Foreign", archive_path="b/Foreign.png"),
        ]
    )


def test_stream_member_hashes_encoded_bytes_and_checks_size_and_crc(tmp_path):
    payload = b"encoded-png-payload"
    info = _info(A.FROZEN_NAMES[0], payload)
    destination = tmp_path / f"{A.FROZEN_NAMES[0]}.png"
    result = A.stream_member(io.BytesIO(payload), info, destination)
    assert destination.read_bytes() == payload
    assert result == {
        "byte_size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "download_crc32": f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
    }
    assert not destination.with_name(destination.name + ".part").exists()

    bad_info = _info(A.FROZEN_NAMES[1], payload)
    bad_info.CRC ^= 1
    bad_destination = tmp_path / f"{A.FROZEN_NAMES[1]}.png"
    with pytest.raises(A.AcquisitionError, match="CRC mismatch"):
        A.stream_member(io.BytesIO(payload), bad_info, bad_destination)
    assert not bad_destination.exists()


class _Archive:
    def __init__(self, infos, payloads):
        self.infos = infos
        self.payloads = payloads
        self.closed = False
        self.opened = []

    def infolist(self):
        return self.infos

    def open(self, info, mode):
        assert mode == "r"
        self.opened.append(PurePosixPath(info.filename).stem)
        return io.BytesIO(self.payloads[PurePosixPath(info.filename).stem])

    def close(self):
        self.closed = True


def test_complete_acquisition_extracts_only_development_and_self_hashes_manifest(tmp_path):
    payloads = {name: b"encoded-" + name.encode() for name in A.FROZEN_NAMES}
    infos = _infos(payloads)
    session = _Session([_head_response(), _head_response()])
    archive = _Archive(infos, payloads)

    def factory(url, *, session, **kwargs):
        assert url == A.FROZEN_URL
        assert kwargs == {
            "initial_buffer_size": 65536,
            "support_suffix_range": True,
            "headers": {"If-Range": '"frozen"', "Accept-Encoding": "identity"},
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
    assert archive.opened == list(A.DEVELOPMENT_NAMES)
    assert {path.name for path in outdir.iterdir()} == {
        *(f"{name}.png" for name in A.DEVELOPMENT_NAMES),
        A.MANIFEST_NAME,
        A.MANIFEST_HASH_NAME,
    }
    for name in A.DEVELOPMENT_NAMES:
        payload = payloads[name]
        assert (outdir / f"{name}.png").read_bytes() == payload
    for name in A.CONFIRMATION_NAMES:
        assert not (outdir / f"{name}.png").exists()

    stored = json.loads((outdir / A.MANIFEST_NAME).read_text())
    unhashed = {key: value for key, value in stored.items() if key != "manifest_payload_sha256"}
    assert stored == manifest
    assert (
        stored["manifest_payload_sha256"] == hashlib.sha256(A.canonical_json(unhashed)).hexdigest()
    )
    assert (
        stored["archive_index_payload_sha256"]
        == hashlib.sha256(A.canonical_json(stored["archive_index"])).hexdigest()
    )
    assert [row["partition"] for row in stored["members"]] == [
        *("development" for _ in A.DEVELOPMENT_NAMES),
        *("confirmation" for _ in A.CONFIRMATION_NAMES),
    ]
    assert [row["payload_extracted"] for row in stored["members"]] == [
        *(True for _ in A.DEVELOPMENT_NAMES),
        *(False for _ in A.CONFIRMATION_NAMES),
    ]
    for row in stored["members"][len(A.DEVELOPMENT_NAMES) :]:
        assert row["local_path"] is None
        assert "sha256" not in row and "byte_size" not in row and "download_crc32" not in row
    process = stored["stage_process"]
    assert process["started_utc"].endswith("Z")
    assert process["ended_utc"].endswith("Z")
    assert process["wall_seconds"] >= 0.0
    assert process["pid"] > 0 and process["parent_pid"] >= 0
    assert Path(process["working_directory"]).is_absolute()
    assert Path(process["python_executable"]).is_absolute()
    assert isinstance(process["invocation_argv"], list) and process["invocation_argv"]
    file_hash = hashlib.sha256((outdir / A.MANIFEST_NAME).read_bytes()).hexdigest()
    assert (outdir / A.MANIFEST_HASH_NAME).read_text() == (f"{file_hash}  {A.MANIFEST_NAME}\n")


def test_acquisition_refuses_nonempty_destination_before_network_and_url_drift(tmp_path):
    outdir = tmp_path / "source"
    outdir.mkdir()
    (outdir / "foreign").write_text("do not overwrite")
    with pytest.raises(A.AcquisitionError, match="must be empty"):
        A.acquire(outdir, session=_Session([]), remote_zip_factory=lambda *args, **kwargs: None)
    with pytest.raises(A.AcquisitionError, match="URL is frozen"):
        A.acquire(
            tmp_path / "source2",
            url="https://example.test/not-clic.zip",
            session=_Session([]),
            remote_zip_factory=lambda *args, **kwargs: None,
        )


def test_cli_has_no_subset_override_and_module_imports_no_image_decoder(tmp_path):
    parser = A.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--outdir", str(tmp_path), "--names", "anything"])
    tree = ast.parse(Path(A.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported.isdisjoint({"PIL", "cv2", "imageio", "torchvision"})
