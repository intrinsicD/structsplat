"""Portable evidence plumbing shared only by HIER-033/034/035 experiments.

Does not select methods or define scientific protocols; those stay in each task and driver.
All outputs are new directories, with an exact file hash manifest after completion.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import itertools
import math
import re
from pathlib import Path
import subprocess
import shutil
import sys

import numpy as np
from PIL import Image


SCHEMA = "structsplat.hier_research.report.v1"
TASKS = {"HIER-033", "HIER-034", "HIER-035"}
TASK_PATHS = {
    "HIER-033": "tasks/HIER-033-pixel-gradient-operator-oracle.md",
    "HIER-034": "tasks/HIER-034-fixed-geometry-basis-cache.md",
    "HIER-035": "tasks/HIER-035-additive-convergence-controls.md",
}
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def protocol_digest(protocol, source_paths):
    source = {name: sha256(ROOT / name) for name in source_paths}
    encoded = json.dumps({"protocol": protocol, "source": source}, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def repository_state():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    status = git("status", "--short")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty": bool(status), "status_sha256": hashlib.sha256(status.encode()).hexdigest()}


def save_rgb(path, value):
    pixels = np.clip(np.asarray(value) * 255 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict, bool)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def has_approved_protocol(task_text, digest):
    for block in task_text.split("### Protocol review")[1:]:
        fields = dict(re.findall(r"#### ([^\n]+)\n([^#]+?)(?=\n#### |\n### |\Z)", block, re.S))
        if (fields.get("Verdict", "").strip() == "Approved"
                and fields.get("Protocol digest", "").strip() == digest
                and fields.get("Outcomes accessed", "").strip() == "No"
                and fields.get("Reviewer", "").strip() not in ("", "codex-root", "pending")):
            return True
    return False


class ResearchBundle:
    def __init__(self, outdir, *, task, protocol, digest, expected_cells, diagnostic=False,
                 source_paths=()):
        if task not in TASKS:
            raise ValueError("unsupported HIER research task")
        self.repository = repository_state()
        if self.repository["dirty"] and not diagnostic:
            raise RuntimeError("formal experiment requires a clean committed source tree")
        if digest != protocol_digest(protocol, source_paths):
            raise ValueError("requested digest differs from current protocol/source")
        if not diagnostic:
            task_text = subprocess.check_output(
                ["git", "show", f'HEAD:{TASK_PATHS[task]}'], cwd=ROOT, text=True)
            if not has_approved_protocol(task_text, digest):
                raise RuntimeError("committed task lacks distinct prospective approval for this digest")
        if len(expected_cells) != len(set(expected_cells)):
            raise ValueError("expected cell identities must be unique")
        self.root = Path(outdir).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        if diagnostic:
            for name in source_paths:
                destination = self.root / "source_snapshot" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, destination)
            diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
            (self.root / "source.diff").write_bytes(diff)
        self.manifest = {
            "schema": SCHEMA, "task": task, "protocol": protocol, "protocol_digest": digest,
            "expected_cells": list(expected_cells), "repository": self.repository,
            "diagnostic": bool(diagnostic), "command": " ".join(sys.argv),
            "source_files": {name: sha256(ROOT / name) for name in source_paths},
            "task_file": TASK_PATHS[task],
        }
        write_json(self.root / "RUNNING.json", self.manifest)

    def finish(self, rows, *, title, interpretation):
        write_json(self.root / "metrics.json", rows)
        with (self.root / "metrics.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        fields = sorted({key for row in rows for key in row})
        with (self.root / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: csv_value(row.get(key)) for key in fields})
        columns = [key for key in ("cell_id", "status", "n_gaussians", "iterations_run",
                   "selected_iteration", "psnr", "ms_ssim", "total_seconds", "cache_bytes")
                   if any(key in row for row in rows)]
        table = "<tr>" + "".join(f"<th>{html.escape(key)}</th>" for key in columns) + "</tr>"
        cards = []
        for row in rows:
            table += "<tr>" + "".join(f"<td>{html.escape(csv_value(row.get(key)))}</td>"
                                      for key in columns) + "</tr>"
            artifacts = row.get("artifacts", {})
            pictures, links = [], []
            for name, path in artifacts.items():
                safe = html.escape(path, quote=True)
                links.append(f'<a href="{safe}">{html.escape(name)}</a>')
                if path.endswith(".png"):
                    pictures.append(f'<figure><a href="{safe}"><img src="{safe}"></a>'
                                    f'<figcaption>{html.escape(name)}</figcaption></figure>')
            error = f'<p class="error">{html.escape(row["error"])}</p>' if row.get("error") else ""
            cards.append(f'<section><h2>{html.escape(row["cell_id"])}</h2>' + error + '<p>'
                         + " · ".join(links) + '</p><div class="images">'
                         + "".join(pictures) + "</div></section>")
        page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>'
                + html.escape(title) + '</title><style>body{font:15px system-ui;margin:32px;'
                'background:#fafafa;color:#152535}table{border-collapse:collapse;font-size:12px}'
                'td,th{padding:6px;border:1px solid #ccd}section{margin-top:28px}'
                '.images{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0}img{max-width:360px;'
                'max-height:300px}a{color:#125fab}figcaption{font-size:12px}</style><h1>'
                + html.escape(title) + '</h1><p>' + html.escape(interpretation)
                + '</p><p><a href="manifest.json">Manifest</a> · <a href="metrics.json">JSON</a>'
                ' · <a href="metrics.jsonl">JSONL</a> · <a href="metrics.csv">CSV</a></p><pre>'
                + html.escape(self.manifest["command"]) + '</pre><table>' + table + '</table>'
                + "".join(cards) + '</html>')
        (self.root / "index.html").write_text(page)
        write_json(self.root / "COMPLETED.json", {"rows": len(rows)})
        # RUNNING.json is retained as the pre-execution identity receipt.
        self.manifest["files"] = {
            path.relative_to(self.root).as_posix(): sha256(path)
            for path in sorted(self.root.rglob("*")) if path.is_file()
        }
        write_json(self.root / "manifest.json", self.manifest)


def _check_protocol_identity(root, manifest, problems):
    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        problems.append("missing source file bindings")
        return
    payload = {"protocol": manifest["protocol"], "source": source_files}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != manifest.get("protocol_digest"):
        problems.append("protocol digest does not match protocol and source bindings")
    initial = json.loads((root / "RUNNING.json").read_text())
    if initial != {key: value for key, value in manifest.items() if key != "files"}:
        problems.append("final manifest identity differs from pre-execution receipt")
    repo = manifest.get("repository", {})
    for name, expected in source_files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            problems.append("source path escapes repository")
            continue
        if manifest.get("diagnostic"):
            path = root / "source_snapshot" / name
            if not path.is_file() or sha256(path) != expected:
                problems.append(f"diagnostic source snapshot disagrees: {name}")
        elif re.fullmatch(r"[a-f0-9]{40}", str(repo.get("commit", ""))):
            try:
                content = subprocess.check_output(["git", "show", f'{repo["commit"]}:{name}'], cwd=ROOT,
                                                  stderr=subprocess.DEVNULL)
                if hashlib.sha256(content).hexdigest() != expected:
                    problems.append(f"source does not match recorded commit: {name}")
            except subprocess.CalledProcessError:
                problems.append(f"source unavailable at recorded commit: {name}")
    if not manifest.get("diagnostic"):
        task_path = TASK_PATHS.get(manifest.get("task"))
        if manifest.get("task_file") != task_path:
            problems.append("unexpected protocol task path")
        try:
            text = subprocess.check_output(["git", "show", f'{repo["commit"]}:{task_path}'], cwd=ROOT,
                                           text=True, stderr=subprocess.DEVNULL)
            if not has_approved_protocol(text, digest):
                problems.append("recorded commit lacks distinct approved protocol digest")
        except (KeyError, subprocess.CalledProcessError):
            problems.append("prospective task receipt unavailable at recorded commit")
    if manifest.get("task") == "HIER-034":
        protocol = manifest["protocol"]
        workloads = [(f, s) for f in protocol["families"] for s in protocol["seeds"]]
        workloads.append((protocol["natural_family"], protocol["natural_seed"]))
        repeats = len(list(itertools.permutations(protocol["backends"])))
        if manifest.get("diagnostic"):
            workloads, repeats = [("smooth", 77)], 1
        expected = {f"{f}_s{s}_r{r}_{b}" for f, s in workloads
                    for r in range(repeats) for b in protocol["backends"]}
        if expected != set(manifest["expected_cells"]) or len(expected) != len(manifest["expected_cells"]):
            problems.append("declared matrix disagrees with frozen executable protocol")
    elif manifest.get("task") == "HIER-035":
        protocol = manifest["protocol"]
        workloads = [(f, s) for f in protocol["families"] for s in protocol["seeds"]]
        if manifest.get("diagnostic"):
            workloads = [("translated", 77)]
        expected = {f"{f}_s{s}_{method}" for f, s in workloads for method in protocol["arms"]}
        if expected != set(manifest["expected_cells"]) or len(expected) != len(manifest["expected_cells"]):
            problems.append("declared matrix disagrees with frozen convergence protocol")
    elif manifest.get("task") == "HIER-033":
        protocol = manifest["protocol"]
        workloads = [(f, s) for f in protocol["families"] for s in protocol["seeds"]]
        if manifest.get("diagnostic"):
            workloads = [("translation", 77)]
        expected = {f"{f}_s{s}_{action}" for f, s in workloads for action in protocol["actions"]}
        if expected != set(manifest["expected_cells"]) or len(expected) != len(manifest["expected_cells"]):
            problems.append("declared matrix disagrees with frozen operator protocol")


def _validate_oracle_cell(root, row, protocol, diagnostic, problems):
    from scripts.check_report_bundle import _finite

    directory = root / "cells" / row["cell_id"]
    config = json.loads((directory / "config.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    if config.get("request") != request or any(
        row.get(key) != request.get(key) for key in ("cell_id", "family", "seed", "action", "smoke")
    ) or any(config.get(key) != row.get(key) for key in ("action_family", "donor", "magnitude", "predicted_gain")):
        problems.append(f"operator row/request/config mismatch: {row['cell_id']}")
    expected = dict(protocol["recovery"])
    expected["steps"] = 2 if diagnostic else expected["steps"]
    if config.get("recovery") != expected:
        problems.append(f"resolved operator recovery differs: {row['cell_id']}")
    history = json.loads((directory / "history.json").read_text())
    trace = history["checkpoints"]
    horizon = expected["steps"]
    if (not isinstance(trace, list) or not trace or not _finite(trace) or len(trace) != horizon + 1
            or [h["iteration"] for h in trace] != list(range(horizon + 1))
            or history["nominal_iterations"] != horizon
            or row["iterations_run"] != row["selected_iteration"] or row["iterations_run"] != horizon
            or row["gradient_evaluations"] != horizon):
        problems.append(f"operator recovery work mismatch: {row['cell_id']}")
        return
    if (any(a["elapsed_seconds"] > b["elapsed_seconds"] for a, b in zip(trace, trace[1:]))
            or trace[-1]["elapsed_seconds"] > row["total_seconds"] + 1e-6):
        problems.append(f"operator temporal trace mismatch: {row['cell_id']}")
    if (row["forward_evaluations"] != horizon + 1
            or trace[-1]["forward_evaluations"] != row["forward_evaluations"]
            or [h["gradient_evaluations"] for h in trace] != list(range(horizon + 1))):
        problems.append(f"operator recovery counters mismatch: {row['cell_id']}")
    ledger = protocol["render_work_accounting"]
    extra = sum(ledger["extra_per_cell_renders"].values())
    if (row.get("counter_scope") != "recovery only"
            or row.get("extra_cell_render_evaluations") != extra
            or row.get("total_cell_render_evaluations") != horizon + 1 + extra
            or row.get("shared_case_render_evaluations") != sum(ledger["shared_per_case_renders"].values())):
        problems.append(f"operator extra-render accounting mismatch: {row['cell_id']}")
    target = np.load(directory / "target.npy", allow_pickle=False)
    for name, key in (("base_reconstruction.npy", "base_objective"),
                      ("immediate_reconstruction.npy", "immediate_objective"),
                      ("reconstruction.npy", "terminal_objective")):
        raw = np.load(directory / name, allow_pickle=False)
        if (list(target.shape) != protocol["shape"] + [3] or raw.shape != target.shape
                or not np.isfinite(raw).all() or not np.isfinite(target).all()
                or np.any(target < 0) or np.any(target > 1)):
            problems.append(f"invalid operator raw image: {row['cell_id']}")
        value = float(0.5 * np.square(raw.astype(np.float64) - target.astype(np.float64)).mean())
        if not math.isclose(value, row[key], rel_tol=1e-6, abs_tol=1e-12):
            problems.append(f"operator raw objective mismatch: {row['cell_id']}")
    if (abs(row["psnr"] + 10 * math.log10(max(2 * row["terminal_objective"], 1e-12))) > 1e-6
            or abs(row["immediate_gain"] - row["base_objective"] + row["immediate_objective"]) > 1e-10
            or abs(row["recovered_gain"] - row["base_objective"] + row["terminal_objective"]) > 1e-10
            or abs(trace[-1]["psnr"] - row["psnr"]) > 1e-3):
        problems.append(f"operator objective/gain/trace mismatch: {row['cell_id']}")
    if (not math.isclose(trace[0]["objective"], row["immediate_objective"], rel_tol=1e-5, abs_tol=1e-12)
            or not math.isclose(trace[-1]["objective"], row["terminal_objective"], rel_tol=1e-5, abs_tol=1e-12)):
        problems.append(f"operator recovery endpoint mismatch: {row['cell_id']}")
    for name in ("base_field", "input_field", "target"):
        extension = ".npy" if name == "target" else ".npz"
        if sha256(directory / (name + extension)) != config[name + "_sha256"]:
            problems.append(f"operator source binding mismatch: {row['cell_id']}")
    fields = []
    n = protocol["n_gaussians"]
    for name in ("base_field.npz", "input_field.npz", "field.npz"):
        with np.load(directory / name, allow_pickle=False) as field:
            values = []
            for key, shape in (("means", (n, 2)), ("log_scales", (n, 2)),
                               ("rotations", (n,)), ("colors", (n, 3))):
                if field[key].shape != shape or not np.isfinite(field[key]).all():
                    problems.append(f"operator serialized count/field mismatch: {row['cell_id']}")
                    return
                values.append(field[key].reshape(n, -1))
            fields.append(np.concatenate(values, 1))
    base, edited, _final = fields
    action = row["action_family"]
    if row["n_gaussians"] != n or action != row["action"].split("_")[0]:
        problems.append(f"operator count/action identity mismatch: {row['cell_id']}")
    if action in ("split", "birth"):
        donor = row["donor"]
        if donor not in protocol["donors"] or f"_d{donor}" not in row["action"]:
            problems.append(f"operator donor identity mismatch: {row['cell_id']}")
        elif action == "birth":
            keep = [i for i in range(n) if i != donor]
            if not np.array_equal(base[keep], edited[keep]):
                problems.append(f"birth changed non-donor rows: {row['cell_id']}")
        elif (not np.array_equal(edited[2], base[2 if donor == 1 else 1])
              or not np.allclose(edited[:2, 5:].sum(0), base[0, 5:], rtol=0, atol=1e-7)
              or not np.array_equal(edited[:2, 2:5], np.repeat(base[0:1, 2:5], 2, axis=0))):
            problems.append(f"split failed donor/color/geometry contract: {row['cell_id']}")
    elif not np.array_equal(base if action == "noop" else base[1:],
                            edited if action == "noop" else edited[1:]):
        problems.append(f"continuous edit changed untouched rows: {row['cell_id']}")
    for key in ("lpips", "cold_render_max_abs", "position_activity", "position_coherence", "proposal_seconds"):
        if isinstance(row.get(key), bool) or not isinstance(row.get(key), (int, float)) or row[key] < 0:
            problems.append(f"invalid operator {key}: {row['cell_id']}")


def _validate_convergence_cell(root, row, protocol, diagnostic, problems):
    from scripts.check_report_bundle import _finite

    directory = root / "cells" / row["cell_id"]
    config = json.loads((directory / "config.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    if config.get("request") != request or any(
        row.get(key) != request.get(key) for key in ("cell_id", "family", "seed", "method", "smoke")
    ):
        problems.append(f"convergence row/request/config mismatch: {row['cell_id']}")
    expected = dict(protocol["config"])
    expected.update(arm="adam" if row["method"].startswith("adam") else row["method"],
                    adam_multiplier=protocol["adam_multipliers"].get(row["method"], 1.0),
                    steps=3 if diagnostic else protocol["config"]["steps"])
    if config.get("control") != expected:
        problems.append(f"resolved convergence config differs: {row['cell_id']}")
    history = json.loads((directory / "history.json").read_text())
    trace = history["checkpoints"]
    horizon = expected["steps"]
    if not isinstance(trace, list) or not trace:
        problems.append(f"missing convergence trace: {row['cell_id']}")
        return
    if (not _finite(trace) or len(trace) != horizon + 1
            or [r["iteration"] for r in trace] != list(range(horizon + 1))
            or history["nominal_iterations"] != horizon
            or row["iterations_run"] != row["selected_iteration"] or row["iterations_run"] != horizon
            or row["gradient_evaluations"] != horizon):
        problems.append(f"convergence terminal/work mismatch: {row['cell_id']}")
    for first, second in zip(trace, trace[1:]):
        if any(first[key] > second[key] for key in
               ("elapsed_seconds", "forward_evaluations", "gradient_evaluations")):
            problems.append(f"convergence trace work/time decreases: {row['cell_id']}")
            break
    if trace[-1]["elapsed_seconds"] > row["total_seconds"] + 1e-6:
        problems.append(f"convergence trace exceeds total time: {row['cell_id']}")
    if (abs(trace[-1]["psnr"] - row["psnr"]) > 1e-3
            or abs(trace[0]["psnr"] - row["initial_psnr"]) > 1e-3):
        problems.append(f"convergence trace/row scoring mismatch: {row['cell_id']}")
    target = np.load(directory / "target.npy", allow_pickle=False)
    raw = np.load(directory / "reconstruction.npy", allow_pickle=False)
    if (list(target.shape) != protocol["shape"] + [3] or raw.shape != target.shape
            or not np.isfinite(target).all() or not np.isfinite(raw).all()
            or np.any(target < 0) or np.any(target > 1)):
        problems.append(f"invalid convergence raw image: {row['cell_id']}")
    mse = float(np.square(raw.astype(np.float64) - target.astype(np.float64)).mean())
    if (not math.isclose(mse, row["raw_mse"], rel_tol=1e-6, abs_tol=1e-12)
            or abs(-10 * math.log10(max(mse, 1e-12)) - row["psnr"]) > 1e-6):
        problems.append(f"convergence raw image/metric mismatch: {row['cell_id']}")
    for name, config_key in (("target.npy", "target_sha256"), ("input_field.npz", "input_field_sha256")):
        if sha256(directory / name) != config[config_key]:
            problems.append(f"convergence input hash mismatch: {row['cell_id']}")
    with np.load(directory / "field.npz", allow_pickle=False) as field:
        n = protocol["n_gaussians"]
        for name, shape in (("means", (n, 2)), ("log_scales", (n, 2)),
                            ("rotations", (n,)), ("colors", (n, 3))):
            if field[name].shape != shape or not np.isfinite(field[name]).all():
                problems.append(f"convergence serialized count/field mismatch: {row['cell_id']}")
    if row["n_gaussians"] != protocol["n_gaussians"]:
        problems.append(f"convergence row count mismatch: {row['cell_id']}")
    for key in ("lpips", "cold_render_max_abs"):
        if isinstance(row.get(key), bool) or not isinstance(row.get(key), (int, float)) or row[key] < 0:
            problems.append(f"invalid convergence {key}: {row['cell_id']}")


def validate_bundle(root, *, allow_dirty=False, allow_error_cells=False):
    """Validate immutable bytes, full matrix, finite rows, links, and clean source identity."""
    from scripts.check_report_bundle import _check_html, _check_repository_identity, _finite

    root = Path(root).resolve()
    problems = []
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        rows = json.loads((root / "metrics.json").read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("task") not in TASKS:
            problems.append("invalid HIER research schema/task")
        _check_repository_identity(manifest.get("repository"), "manifest", problems,
                                   allow_dirty=allow_dirty)
        if manifest.get("diagnostic") and not allow_dirty:
            problems.append("diagnostic experiment is not formal evidence")
        _check_protocol_identity(root, manifest, problems)
        if not isinstance(manifest.get("protocol_digest"), str) or len(manifest["protocol_digest"]) != 64:
            problems.append("missing protocol digest")
        if not isinstance(rows, list) or not rows or not _finite(rows):
            return problems + ["missing or nonfinite metric rows"]
        ids = [row["cell_id"] for row in rows]
        expected = manifest["expected_cells"]
        if len(ids) != len(set(ids)) or sorted(ids) != sorted(expected):
            problems.append("metric rows differ from frozen complete cell matrix")
        if [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()] != rows:
            problems.append("JSON and JSONL rows disagree")
        with (root / "metrics.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = sorted({key for row in rows for key in row})
            if reader.fieldnames != fields or list(reader) != [
                {key: csv_value(row.get(key)) for key in fields} for row in rows
            ]:
                problems.append("CSV projection disagrees with JSON")
        actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*")
                        if p.is_file() and p.name != "manifest.json"}
        files = manifest.get("files", {})
        if actual_files != set(files):
            problems.append("manifest does not enumerate the exact artifact file set")
        for name, digest in files.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
                problems.append(f"missing, escaped, or changed artifact: {name}")
        linked = _check_html(root, problems)
        for row in rows:
            if row.get("status") not in ("ok", "error"):
                problems.append(f"invalid cell status: {row['cell_id']}")
            if row.get("status") == "error":
                if not row.get("error"):
                    problems.append("error cell lacks error explanation")
                if not allow_error_cells:
                    problems.append(f"failed cell: {row['cell_id']}")
            if row.get("status") == "ok":
                required = {"field.npz", "history.json", "config.json", "target.png",
                            "reconstruction.png", "error.png", "curves.png"}
                if not required.issubset(row.get("artifacts", {})):
                    problems.append(f"missing success artifacts: {row['cell_id']}")
                for key in ("psnr", "ms_ssim", "total_seconds"):
                    value = row.get(key)
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        problems.append(f"missing numeric {key}: {row['cell_id']}")
                for key in ("n_gaussians", "iterations_run", "selected_iteration"):
                    value = row.get(key)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        problems.append(f"invalid {key}: {row['cell_id']}")
                if row.get("total_seconds", 0) <= 0 or row.get("n_gaussians", 0) < 1:
                    problems.append(f"nonpositive time/count: {row['cell_id']}")
                if manifest["task"] == "HIER-034":
                    directory = root / "cells" / row["cell_id"]
                    config = json.loads((directory / "config.json").read_text())
                    request = json.loads((directory / "request.json").read_text())
                    if config.get("request") != request or any(
                        request.get(key) != row.get(key) for key in ("cell_id", "family", "seed", "repeat", "backend")
                    ):
                        problems.append(f"row/request/config mismatch: {row['cell_id']}")
                    for key in ("cold_render_max_abs", "maintained_parity_max_abs", "raw_sse", "cache_bytes"):
                        if isinstance(row.get(key), bool) or not isinstance(row.get(key), (int, float)) or row[key] < 0:
                            problems.append(f"invalid {key}: {row['cell_id']}")
                elif manifest["task"] == "HIER-035":
                    _validate_convergence_cell(root, row, manifest["protocol"], manifest["diagnostic"], problems)
                elif manifest["task"] == "HIER-033":
                    _validate_oracle_cell(root, row, manifest["protocol"], manifest["diagnostic"], problems)
            for name in row.get("artifacts", {}).values():
                if name not in files or (root / name).resolve() not in linked:
                    problems.append(f"unexposed or unhashed artifact: {name}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(f"malformed HIER research bundle: {exc}")
    return problems
