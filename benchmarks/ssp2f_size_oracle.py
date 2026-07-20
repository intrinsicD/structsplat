"""Exact complete-byte SSP2F oracle for scalar absolute-RGB edits.

The oracle keeps SSP2F geometry and all non-RGB symbols frozen.  A proposal
recomputes the edited channel's complete empirical normalization, its
order-sensitive arithmetic byte count through the native repeated-CDF ABI,
and the canonical zlib-9 member for the entire five-channel model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmarks import ssp2e_container as container


RGB_CHANNELS = ("red", "green", "blue")
NON_RGB_CHANNELS = ("mean_x", "mean_y", "scale_x", "scale_y", "rotation")
_STATE_DOMAIN = b"structsplat-ssp2f-size-oracle-state-v1\0"
_PROPOSAL_DOMAIN = b"structsplat-ssp2f-size-oracle-proposal-v1\0"


class SSP2FSizeOracleError(RuntimeError):
    """An exact SSP2F size-state invariant was violated."""


@dataclass(frozen=True)
class SSP2FSizeProposal:
    """One exact scalar edit evaluated against a particular oracle state."""

    epoch: int
    state_sha256: str
    row: int
    channel: str
    old_symbol: int
    new_symbol: int
    frequency_payload: bytes
    arithmetic_bytes: int
    model_payload: bytes
    complete_bytes: int
    proposal_sha256: str


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _cdf_row(frequencies: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate((np.zeros(1, dtype=np.uint32), np.cumsum(frequencies, dtype=np.uint32)))
    )


def _validated_rgb(
    parts: container.SSPL1Parts,
    rgb_symbols: np.ndarray | None,
) -> np.ndarray:
    if rgb_symbols is None:
        return np.ascontiguousarray(
            np.column_stack(tuple(parts.symbols[name] for name in RGB_CHANNELS)),
            dtype=np.uint32,
        )
    rgb = np.asarray(rgb_symbols)
    alphabet = 1 << parts.header.bits[3]
    if (
        rgb.shape != (parts.header.n, 3)
        or rgb.dtype.kind not in "iu"
        or (rgb.dtype.kind == "i" and np.any(rgb < 0))
        or (rgb.size and int(rgb.max()) >= alphabet)
    ):
        raise SSP2FSizeOracleError(
            f"RGB state must be an integer [{parts.header.n},3] array inside [0,{alphabet})"
        )
    return np.ascontiguousarray(rgb, dtype=np.uint32)


class SSP2FSizeOracle:
    """Stateful exact complete-byte oracle for one frozen SSP2F source field."""

    def __init__(
        self,
        sspl1_blob: bytes,
        native_coder: Any,
        *,
        rgb_symbols: np.ndarray | None = None,
    ) -> None:
        for name in ("encode", "decode", "size_repeated"):
            if not callable(getattr(native_coder, name, None)):
                raise SSP2FSizeOracleError(
                    "native_coder must expose encode, decode, and size_repeated"
                )
        self._source_blob = sspl1_blob
        self._parts = container.extract_sspl1_parts(sspl1_blob)
        self._native = native_coder
        rgb = _validated_rgb(self._parts, rgb_symbols)
        self._symbols = {
            name: np.ascontiguousarray(values, dtype=np.uint32).copy()
            for name, values in self._parts.symbols.items()
        }
        for index, name in enumerate(RGB_CHANNELS):
            self._symbols[name] = np.ascontiguousarray(rgb[:, index]).copy()

        self._counts: dict[str, np.ndarray] = {}
        self._frequencies: dict[str, np.ndarray] = {}
        self._arithmetic_bytes: dict[str, int] = {}
        for name in container.MODELED_CHANNELS:
            alphabet = self._channel_alphabet(name)
            counts = np.bincount(
                self._symbols[name].astype(np.int64),
                minlength=alphabet,
            ).astype(np.uint64)
            frequencies = container.empirical_frequencies_from_counts(counts, alphabet)
            self._counts[name] = np.ascontiguousarray(counts)
            self._frequencies[name] = frequencies
            self._arithmetic_bytes[name] = int(
                self._native.size_repeated(self._symbols[name], _cdf_row(frequencies))
            )

        self._scratch = {
            name: np.empty(self._parts.header.n, dtype=np.uint32) for name in RGB_CHANNELS
        }
        self._model_payload = container.serialize_empirical_frequencies(
            self._frequencies,
            self._parts.header,
        )
        self._fixed_bytes = (
            container.FACTORIZED_PREFIX_BYTES
            + len(self._parts.payloads["means"])
            + len(self._parts.payloads["rotation"])
        )
        self._complete_bytes = (
            self._fixed_bytes + sum(self._arithmetic_bytes.values()) + len(self._model_payload)
        )
        self._epoch = 0
        self._state_sha256 = self._compute_state_sha256()
        self._assert_state()

    def _channel_alphabet(self, name: str) -> int:
        if name in ("scale_x", "scale_y"):
            return 1 << self._parts.header.bits[1]
        if name in RGB_CHANNELS:
            return 1 << self._parts.header.bits[3]
        raise SSP2FSizeOracleError(f"{name!r} is not an SSP2F arithmetic channel")

    def _compute_state_sha256(self) -> str:
        digest = hashlib.sha256(_STATE_DOMAIN)
        digest.update(self._parts.blob_sha256.encode("ascii"))
        for name in RGB_CHANNELS:
            digest.update(name.encode("ascii") + b"\0")
            digest.update(self._symbols[name].astype("<u4", copy=False).tobytes(order="C"))
        for name in container.MODELED_CHANNELS:
            digest.update(name.encode("ascii") + b"\0")
            digest.update(self._counts[name].astype("<u8", copy=False).tobytes(order="C"))
            digest.update(self._frequencies[name].astype("<u4", copy=False).tobytes(order="C"))
            digest.update(int(self._arithmetic_bytes[name]).to_bytes(8, "little"))
        digest.update(self._model_payload)
        digest.update(int(self._complete_bytes).to_bytes(8, "little"))
        return digest.hexdigest()

    @staticmethod
    def _proposal_digest(
        *,
        epoch: int,
        state_sha256: str,
        row: int,
        channel: str,
        old_symbol: int,
        new_symbol: int,
        frequency_payload: bytes,
        arithmetic_bytes: int,
        model_payload: bytes,
        complete_bytes: int,
    ) -> str:
        record = {
            "epoch": epoch,
            "state_sha256": state_sha256,
            "row": row,
            "channel": channel,
            "old_symbol": old_symbol,
            "new_symbol": new_symbol,
            "frequency_sha256": hashlib.sha256(frequency_payload).hexdigest(),
            "frequency_bytes": len(frequency_payload),
            "arithmetic_bytes": arithmetic_bytes,
            "model_sha256": hashlib.sha256(model_payload).hexdigest(),
            "model_bytes": len(model_payload),
            "complete_bytes": complete_bytes,
        }
        return hashlib.sha256(_PROPOSAL_DOMAIN + _canonical_json(record)).hexdigest()

    def _assert_state(self) -> None:
        for name in container.MODELED_CHANNELS:
            alphabet = self._channel_alphabet(name)
            observed_counts = np.bincount(
                self._symbols[name].astype(np.int64),
                minlength=alphabet,
            ).astype(np.uint64)
            if not np.array_equal(observed_counts, self._counts[name]):
                raise SSP2FSizeOracleError(f"cached histogram drift for {name}")
            normalized = container.empirical_frequencies_from_counts(
                self._counts[name],
                alphabet,
            )
            if not np.array_equal(normalized, self._frequencies[name]):
                raise SSP2FSizeOracleError(f"cached empirical frequencies drift for {name}")
            if self._arithmetic_bytes[name] <= 0:
                raise SSP2FSizeOracleError(f"nonpositive arithmetic size for {name}")
        expected_model = container.serialize_empirical_frequencies(
            self._frequencies,
            self._parts.header,
        )
        if expected_model != self._model_payload:
            raise SSP2FSizeOracleError("cached complete empirical model drift")
        expected_complete = (
            self._fixed_bytes + sum(self._arithmetic_bytes.values()) + len(self._model_payload)
        )
        if expected_complete != self._complete_bytes:
            raise SSP2FSizeOracleError("complete SSP2F byte accounting drift")
        if self._compute_state_sha256() != self._state_sha256:
            raise SSP2FSizeOracleError("SSP2F oracle state digest drift")

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def state_sha256(self) -> str:
        return self._state_sha256

    @property
    def complete_bytes(self) -> int:
        return self._complete_bytes

    @property
    def arithmetic_bytes(self) -> dict[str, int]:
        return dict(self._arithmetic_bytes)

    @property
    def model_payload_sha256(self) -> str:
        return hashlib.sha256(self._model_payload).hexdigest()

    @property
    def rgb_symbols(self) -> np.ndarray:
        return np.ascontiguousarray(
            np.column_stack(tuple(self._symbols[name] for name in RGB_CHANNELS)),
            dtype=np.uint32,
        )

    @property
    def frequencies(self) -> dict[str, np.ndarray]:
        return {name: values.copy() for name, values in self._frequencies.items()}

    @property
    def byte_breakdown(self) -> dict[str, int]:
        result = {
            "prefix": container.FACTORIZED_PREFIX_BYTES,
            "means": len(self._parts.payloads["means"]),
            "scale_x_arithmetic": self._arithmetic_bytes["scale_x"],
            "scale_y_arithmetic": self._arithmetic_bytes["scale_y"],
            "rotation": len(self._parts.payloads["rotation"]),
            "red_arithmetic": self._arithmetic_bytes["red"],
            "green_arithmetic": self._arithmetic_bytes["green"],
            "blue_arithmetic": self._arithmetic_bytes["blue"],
            "model_zlib9": len(self._model_payload),
        }
        if sum(result.values()) != self._complete_bytes:
            raise SSP2FSizeOracleError("reader-facing byte breakdown does not sum exactly")
        return result

    def propose(self, row: int, channel: str, new_symbol: int) -> SSP2FSizeProposal:
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)):
            raise SSP2FSizeOracleError("proposal row must be an integer")
        normalized_row = int(row)
        if not 0 <= normalized_row < self._parts.header.n:
            raise SSP2FSizeOracleError("proposal row is out of range")
        if channel not in RGB_CHANNELS:
            raise SSP2FSizeOracleError("proposal channel must be red, green, or blue")
        if isinstance(new_symbol, bool) or not isinstance(new_symbol, (int, np.integer)):
            raise SSP2FSizeOracleError("proposal symbol must be an integer")
        normalized_symbol = int(new_symbol)
        alphabet = self._channel_alphabet(channel)
        if not 0 <= normalized_symbol < alphabet:
            raise SSP2FSizeOracleError("proposal symbol is outside the RGB alphabet")
        old_symbol = int(self._symbols[channel][normalized_row])
        if old_symbol == normalized_symbol:
            raise SSP2FSizeOracleError("proposal must change the selected symbol")

        counts = self._counts[channel].copy()
        counts[old_symbol] -= 1
        counts[normalized_symbol] += 1
        frequencies = container.empirical_frequencies_from_counts(counts, alphabet)
        scratch = self._scratch[channel]
        np.copyto(scratch, self._symbols[channel])
        scratch[normalized_row] = normalized_symbol
        arithmetic_bytes = int(self._native.size_repeated(scratch, _cdf_row(frequencies)))
        candidate_frequencies = dict(self._frequencies)
        candidate_frequencies[channel] = frequencies
        model_payload = container.serialize_empirical_frequencies(
            candidate_frequencies,
            self._parts.header,
        )
        complete_bytes = (
            self._complete_bytes
            - self._arithmetic_bytes[channel]
            - len(self._model_payload)
            + arithmetic_bytes
            + len(model_payload)
        )
        frequency_payload = frequencies.astype("<u4", copy=False).tobytes(order="C")
        proposal_sha256 = self._proposal_digest(
            epoch=self._epoch,
            state_sha256=self._state_sha256,
            row=normalized_row,
            channel=channel,
            old_symbol=old_symbol,
            new_symbol=normalized_symbol,
            frequency_payload=frequency_payload,
            arithmetic_bytes=arithmetic_bytes,
            model_payload=model_payload,
            complete_bytes=complete_bytes,
        )
        return SSP2FSizeProposal(
            epoch=self._epoch,
            state_sha256=self._state_sha256,
            row=normalized_row,
            channel=channel,
            old_symbol=old_symbol,
            new_symbol=normalized_symbol,
            frequency_payload=frequency_payload,
            arithmetic_bytes=arithmetic_bytes,
            model_payload=model_payload,
            complete_bytes=complete_bytes,
            proposal_sha256=proposal_sha256,
        )

    def commit(self, proposal: SSP2FSizeProposal) -> int:
        if not isinstance(proposal, SSP2FSizeProposal):
            raise SSP2FSizeOracleError("commit requires an SSP2FSizeProposal")
        if proposal.epoch != self._epoch or proposal.state_sha256 != self._state_sha256:
            raise SSP2FSizeOracleError("stale SSP2F size proposal")
        expected = self.propose(proposal.row, proposal.channel, proposal.new_symbol)
        if expected != proposal:
            raise SSP2FSizeOracleError("SSP2F size proposal payload or digest differs")

        frequencies = np.frombuffer(proposal.frequency_payload, dtype="<u4").astype(np.uint32)
        old_symbol = int(self._symbols[proposal.channel][proposal.row])
        self._symbols[proposal.channel][proposal.row] = proposal.new_symbol
        self._counts[proposal.channel][old_symbol] -= 1
        self._counts[proposal.channel][proposal.new_symbol] += 1
        self._frequencies[proposal.channel] = np.ascontiguousarray(frequencies)
        self._arithmetic_bytes[proposal.channel] = proposal.arithmetic_bytes
        self._model_payload = proposal.model_payload
        self._complete_bytes = proposal.complete_bytes
        self._epoch += 1
        self._state_sha256 = self._compute_state_sha256()
        self._assert_state()
        return self._complete_bytes

    def encode_and_verify(self) -> bytes:
        """Full-encode, cold-decode, re-encode, and match every cached byte component."""
        self._assert_state()
        blob = container.encode_ssp2f_rgb_override(
            self._source_blob,
            self.rgb_symbols,
            arithmetic_encode=self._native.encode,
            arithmetic_decode=self._native.decode,
        )
        parsed = container.parse_container(blob, reference_sspl1=self._source_blob)
        if parsed.prefix_bytes != container.FACTORIZED_PREFIX_BYTES:
            raise SSP2FSizeOracleError("SSP2F factorized prefix drift")
        if parsed.payloads["means"] != self._parts.payloads["means"]:
            raise SSP2FSizeOracleError("SSP2F oracle changed the means payload")
        if parsed.payloads["rotation"] != self._parts.payloads["rotation"]:
            raise SSP2FSizeOracleError("SSP2F oracle changed the rotation payload")
        for name in container.MODELED_CHANNELS:
            if len(parsed.payloads[name]) != self._arithmetic_bytes[name]:
                raise SSP2FSizeOracleError(f"oracle/full-encode arithmetic size differs for {name}")
        if parsed.payloads["model"] != self._model_payload:
            raise SSP2FSizeOracleError("oracle/full-encode complete model bytes differ")
        if len(blob) != self._complete_bytes:
            raise SSP2FSizeOracleError(
                f"oracle/full-encode complete bytes differ: {self._complete_bytes} != {len(blob)}"
            )
        if sum(self.byte_breakdown.values()) != len(blob):
            raise SSP2FSizeOracleError("full SSP2F formula does not equal physical bytes")
        return blob


__all__ = [
    "RGB_CHANNELS",
    "SSP2FSizeOracle",
    "SSP2FSizeOracleError",
    "SSP2FSizeProposal",
]
