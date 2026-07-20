"""Generate and strictly load the frozen SSP2E v1 probability assets.

The two binary assets in :mod:`benchmarks.assets` are decoder source state, not
per-image side information.  Generation intentionally uses only ``Decimal``
operations under the context frozen by ``tasks/COMP-009-ssp2e-actual-coder.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
import tempfile
from array import array
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

CDF_MAGIC = b"SSP2ECDF"
COST_MAGIC = b"SSP2ECST"
ASSET_VERSION = 1
TOTAL_FREQUENCY = 32768
COST_Q = 1 << 24
ALPHABETS = (64, 256)
LOCATION_COUNT = 64
SCALE_Q8: tuple[int, ...] = (
    64,
    93,
    134,
    194,
    281,
    406,
    588,
    851,
    1232,
    1783,
    2580,
    3734,
    5405,
    7822,
    11321,
    16384,
)

ASSET_DIRECTORY = Path(__file__).resolve().with_name("assets")
CDF_ASSET_NAME = "ssp2e_cdf_v1.bin"
COST_ASSET_NAME = "ssp2e_cost_q24_v1.bin"
CDF_ASSET_SIZE = 655_388
COST_ASSET_SIZE = 16 + 4 * TOTAL_FREQUENCY
CDF_ASSET_SHA256 = "89534aa31405e5129fc57727076dfc32829deb8090362e43869110db2a45c493"
COST_ASSET_SHA256 = "b67d48caf7c35a79a78663fba83ce63d1df5d892efaf43c921d056d8fdf65c7e"


class AssetError(ValueError):
    """A frozen asset is absent, malformed, or does not match its protocol hash."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _frozen_decimal_context():
    return localcontext()


def _frequency_row(alphabet: int, location: int, scale_q8: int) -> tuple[int, ...]:
    with _frozen_decimal_context() as context:
        context.prec = 160
        context.rounding = ROUND_HALF_EVEN
        one = Decimal(1)
        half = Decimal(1) / Decimal(2)
        mu = Decimal(location) * Decimal(alphabet - 1) / Decimal(63)
        sigma = Decimal(scale_q8) / Decimal(256)
        boundary = [Decimal(0)]
        for k in range(1, alphabet):
            exponent = -(Decimal(k) - half - mu) / sigma
            boundary.append(one / (one + exponent.exp()))
        boundary.append(one)

        residual = TOTAL_FREQUENCY - alphabet
        quotas = [
            Decimal(residual) * (boundary[symbol + 1] - boundary[symbol])
            for symbol in range(alphabet)
        ]
        allocated = [int(quota.to_integral_value(rounding=ROUND_FLOOR)) for quota in quotas]
        remaining = residual - sum(allocated)
        order = sorted(
            range(alphabet),
            key=lambda symbol: (-(quotas[symbol] - Decimal(allocated[symbol])), symbol),
        )
        for symbol in order[:remaining]:
            allocated[symbol] += 1
        frequencies = tuple(1 + value for value in allocated)

    if min(frequencies) <= 0 or sum(frequencies) != TOTAL_FREQUENCY:
        raise AssertionError("Decimal allocation did not produce a positive exact composition")
    return frequencies


def build_cdf_asset_bytes() -> bytes:
    """Return the exact universal SSP2E CDF asset serialization."""

    output = bytearray(struct.pack("<8sIII", CDF_MAGIC, ASSET_VERSION, TOTAL_FREQUENCY, 2))
    for alphabet in ALPHABETS:
        output.extend(struct.pack("<HH", alphabet, 0))
        for location in range(LOCATION_COUNT):
            for scale_q8 in SCALE_Q8:
                output.extend(
                    struct.pack(
                        f"<{alphabet}H",
                        *_frequency_row(alphabet, location, scale_q8),
                    )
                )
    result = bytes(output)
    if len(result) != CDF_ASSET_SIZE:
        raise AssertionError(f"CDF asset has {len(result)} bytes, expected {CDF_ASSET_SIZE}")
    return result


def build_cost_asset_bytes() -> bytes:
    """Return the exact encoder-only Q24 fitting-cost asset serialization."""

    output = bytearray(struct.pack("<8sII", COST_MAGIC, ASSET_VERSION, COST_Q))
    with _frozen_decimal_context() as context:
        context.prec = 160
        context.rounding = ROUND_HALF_EVEN
        two_log = Decimal(2).ln()
        total = Decimal(TOTAL_FREQUENCY)
        for frequency in range(1, TOTAL_FREQUENCY + 1):
            log2_cost = (total / Decimal(frequency)).ln() / two_log
            cost = int((Decimal(COST_Q) * log2_cost).to_integral_value(rounding=ROUND_HALF_EVEN))
            output.extend(struct.pack("<I", cost))
    result = bytes(output)
    if len(result) != COST_ASSET_SIZE:
        raise AssertionError(f"cost asset has {len(result)} bytes, expected {COST_ASSET_SIZE}")
    return result


def _write_atomic_exact(path: Path, data: bytes, expected_hash: str) -> None:
    observed = _sha256(data)
    if observed != expected_hash:
        raise AssetError(
            f"refusing to write {path.name}: generated SHA-256 {observed}, expected {expected_hash}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def generate_assets(directory: Path = ASSET_DIRECTORY) -> tuple[Path, Path]:
    """Generate both assets, verify their frozen hashes, and atomically install them."""

    cdf_path = directory / CDF_ASSET_NAME
    cost_path = directory / COST_ASSET_NAME
    _write_atomic_exact(cdf_path, build_cdf_asset_bytes(), CDF_ASSET_SHA256)
    _write_atomic_exact(cost_path, build_cost_asset_bytes(), COST_ASSET_SHA256)
    return cdf_path, cost_path


def _read_exact(path: Path, expected_size: int, expected_hash: str) -> bytes:
    try:
        data = path.read_bytes()
    except FileNotFoundError as error:
        raise AssetError(f"missing frozen asset: {path}") from error
    if len(data) != expected_size:
        raise AssetError(f"{path} has {len(data)} bytes, expected {expected_size}")
    observed = _sha256(data)
    if observed != expected_hash:
        raise AssetError(f"{path} SHA-256 is {observed}, expected {expected_hash}")
    return data


@dataclass(frozen=True)
class StaticCDFTable:
    """One alphabet's 64-by-16 collection of strictly positive CDF rows."""

    alphabet: int
    frequencies: tuple[int, ...]
    cdfs: tuple[tuple[int, ...], ...]

    def row(self, location: int, scale_index: int) -> tuple[int, ...]:
        if not 0 <= location < LOCATION_COUNT:
            raise IndexError(f"location {location} is outside [0, {LOCATION_COUNT})")
        if not 0 <= scale_index < len(SCALE_Q8):
            raise IndexError(f"scale index {scale_index} is outside [0, {len(SCALE_Q8)})")
        return self.cdfs[location * len(SCALE_Q8) + scale_index]

    def frequency_row(self, location: int, scale_index: int) -> tuple[int, ...]:
        offset = (location * len(SCALE_Q8) + scale_index) * self.alphabet
        return self.frequencies[offset : offset + self.alphabet]


def load_tables(path: Path | None = None) -> Mapping[int, StaticCDFTable]:
    """Strictly verify and parse the frozen universal CDF asset."""

    asset_path = ASSET_DIRECTORY / CDF_ASSET_NAME if path is None else Path(path)
    data = _read_exact(asset_path, CDF_ASSET_SIZE, CDF_ASSET_SHA256)
    magic, version, total, table_count = struct.unpack_from("<8sIII", data, 0)
    if (magic, version, total, table_count) != (CDF_MAGIC, ASSET_VERSION, TOTAL_FREQUENCY, 2):
        raise AssetError("CDF asset header is invalid")
    cursor = struct.calcsize("<8sIII")
    tables: dict[int, StaticCDFTable] = {}
    for expected_alphabet in ALPHABETS:
        alphabet, reserved = struct.unpack_from("<HH", data, cursor)
        cursor += 4
        if alphabet != expected_alphabet or reserved != 0 or alphabet in tables:
            raise AssetError("CDF asset table header/order is invalid")
        count = LOCATION_COUNT * len(SCALE_Q8) * alphabet
        byte_count = 2 * count
        packed = array("H")
        packed.frombytes(data[cursor : cursor + byte_count])
        if sys.byteorder != "little":
            packed.byteswap()
        frequencies = tuple(int(value) for value in packed)
        cursor += byte_count
        cdfs: list[tuple[int, ...]] = []
        for row_offset in range(0, count, alphabet):
            cumulative = [0]
            for frequency in frequencies[row_offset : row_offset + alphabet]:
                if frequency <= 0:
                    raise AssetError("CDF asset contains a nonpositive frequency")
                cumulative.append(cumulative[-1] + frequency)
            if cumulative[-1] != TOTAL_FREQUENCY:
                raise AssetError("CDF asset row does not sum to the frozen total")
            cdfs.append(tuple(cumulative))
        tables[alphabet] = StaticCDFTable(alphabet, frequencies, tuple(cdfs))
    if cursor != len(data):
        raise AssetError("CDF asset has a trailing suffix")
    return MappingProxyType(tables)


def load_costs(path: Path | None = None) -> tuple[int, ...]:
    """Strictly verify and parse the frozen Q24 cost array, indexed by frequency."""

    asset_path = ASSET_DIRECTORY / COST_ASSET_NAME if path is None else Path(path)
    data = _read_exact(asset_path, COST_ASSET_SIZE, COST_ASSET_SHA256)
    magic, version, q_value = struct.unpack_from("<8sII", data, 0)
    if (magic, version, q_value) != (COST_MAGIC, ASSET_VERSION, COST_Q):
        raise AssetError("cost asset header is invalid")
    packed = array("I")
    packed.frombytes(data[struct.calcsize("<8sII") :])
    if sys.byteorder != "little":
        packed.byteswap()
    if len(packed) != TOTAL_FREQUENCY:
        raise AssetError("cost asset entry count is invalid")
    # Index zero is intentionally a sentinel: valid frequencies are 1..32768.
    return (0, *(int(value) for value in packed))


def verify_assets(directory: Path = ASSET_DIRECTORY) -> dict[str, object]:
    """Load both assets and return their immutable identity surface."""

    tables = load_tables(directory / CDF_ASSET_NAME)
    costs = load_costs(directory / COST_ASSET_NAME)
    return {
        "cdf_path": str(directory / CDF_ASSET_NAME),
        "cdf_sha256": CDF_ASSET_SHA256,
        "cdf_size": CDF_ASSET_SIZE,
        "cost_path": str(directory / COST_ASSET_NAME),
        "cost_sha256": COST_ASSET_SHA256,
        "cost_size": COST_ASSET_SIZE,
        "alphabets": list(tables),
        "cost_count": len(costs) - 1,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "verify"))
    parser.add_argument("--directory", type=Path, default=ASSET_DIRECTORY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "generate":
        generate_assets(args.directory)
    identity = verify_assets(args.directory)
    for key, value in identity.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
