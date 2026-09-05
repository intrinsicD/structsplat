"""Portable evidence plumbing for explicitly registered bounded research experiments.

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
TASKS = {"HIER-033", "HIER-034", "HIER-035", "HIER-036", "FIT-050", "PORT-007", "FIT-051"}
TASK_PATHS = {
    "HIER-033": "tasks/HIER-033-pixel-gradient-operator-oracle.md",
    "HIER-034": "tasks/HIER-034-fixed-geometry-basis-cache.md",
    "HIER-035": "tasks/HIER-035-additive-convergence-controls.md",
    "HIER-036": "tasks/HIER-036-dense-coupling-oracle.md",
    "FIT-050": "tasks/FIT-050-safe-color-ray.md",
    "PORT-007": "tasks/PORT-007-joint-render-coverage.md",
    "FIT-051": "tasks/FIT-051-actual-render-color-ray.md",
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
        columns = [key for key in ("cell_id", "method", "image_id", "image", "seed", "status",
                   "n_gaussians", "iterations_run", "selected_iteration", "psnr", "ms_ssim",
                   "lpips", "total_seconds", "cache_bytes")
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
        extra_links = "".join(
            f' · <a href="{name}">{label}</a>'
            for name, label in (("summary.json", "Frozen gate summary"),
                                ("decision.json", "Frozen gate decision"),
                                ("occupancy.json", "Resource observations"),
                                ("parent_source.json", "Parent provenance"))
            if (self.root / name).is_file()
        )
        page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>'
                + html.escape(title) + '</title><style>body{font:15px system-ui;margin:32px;'
                'background:#fafafa;color:#152535}table{border-collapse:collapse;font-size:12px}'
                'td,th{padding:6px;border:1px solid #ccd}section{margin-top:28px}'
                '.images{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0}img{max-width:360px;'
                'max-height:300px}a{color:#125fab}figcaption{font-size:12px}</style><h1>'
                + html.escape(title) + '</h1><p>' + html.escape(interpretation)
                + '</p><p><a href="manifest.json">Manifest</a> · <a href="metrics.json">JSON</a>'
                ' · <a href="metrics.jsonl">JSONL</a> · <a href="metrics.csv">CSV</a>'
                + extra_links + '</p><pre>'
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
    elif manifest.get("task") in ("HIER-035", "HIER-036"):
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
    if protocol["task"] == "HIER-036" and not row["method"].startswith("adam"):
        expected["arm"] = "block"
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


def _validate_coupling_cell(root, row, protocol, diagnostic, problems):
    """HIER-036's factorial configuration, bounds, ceiling, and complete inner-work contract."""
    _validate_convergence_cell(root, row, protocol, diagnostic, problems)
    directory = root / "cells" / row["cell_id"]
    config = json.loads((directory / "config.json").read_text())
    method = row["method"]
    curvature = not method.startswith("adam")
    if (config.get("dense_mode") != (method if curvature else None)
            or config.get("dense_limits") != protocol["dense_limits"]
            or config.get("precision") != protocol["precision"]):
        problems.append(f"coupling mode/limits/precision mismatch: {row['cell_id']}")
    stratum = next((k for k, seeds in protocol["strata"].items() if row["seed"] in seeds), "diagnostic")
    if row.get("stratum") != stratum:
        problems.append(f"coupling exposure stratum mismatch: {row['cell_id']}")
    trace = json.loads((directory / "history.json").read_text())["checkpoints"]
    progress = [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()]
    if progress != trace:
        problems.append(f"coupling progress/history mismatch: {row['cell_id']}")
    horizon = 3 if diagnostic else protocol["config"]["steps"]
    forwards = 1
    for step, item in enumerate(trace):
        trials = item["line_search_trials"]
        if (item["gradient_evaluations"] != step
                or not isinstance(item["accepted"], bool)
                or (step == 0 and trials != 0)
                or (step > 0 and curvature and not 1 <= trials <= protocol["config"]["max_backtracks"])
                or (not curvature and (trials != 0 or not item["accepted"]))):
            problems.append(f"coupling trace gradient/trials/acceptance mismatch: {row['cell_id']}")
        if step:
            forwards += trials if curvature else 1
        if item["forward_evaluations"] != forwards:
            problems.append(f"coupling forward work mismatch: {row['cell_id']}")
        if curvature:
            if item["jacobian_constructions"] != step or item["linear_solves"] != step:
                problems.append(f"coupling dense inner-work mismatch: {row['cell_id']}")
            if step and (item["objective"] > trace[step - 1]["objective"]
                         or (not item["accepted"] and
                             item["objective"] != trace[step - 1]["objective"])):
                problems.append(f"coupling finite-acceptance violation: {row['cell_id']}")
    if (row["forward_evaluations"] != forwards
            or row["jacobian_constructions"] != (horizon if curvature else 0)
            or row["linear_solves"] != (horizon if curvature else 0)
            or row["rejected_updates"] != sum(not r["accepted"] for r in trace[1:])):
        problems.append(f"coupling terminal work mismatch: {row['cell_id']}")
    n = protocol["n_gaussians"]
    j_bytes = math.prod(protocol["shape"]) * 3 * n * 8 * 4 if curvature else 0
    gram_bytes = (n * 8) ** 2 * 4 if curvature else 0
    if (row["retained_jacobian_bytes"] != j_bytes or row["retained_gram_bytes"] != gram_bytes
            or j_bytes > protocol["dense_limits"]["max_jacobian_bytes"]
            or row["peak_allocated_bytes"] < j_bytes):
        problems.append(f"coupling retained/peak memory mismatch: {row['cell_id']}")
    warm = row["warmup_forward_evaluations"]
    if (not 21 <= warm <= 61
            or row["warmup_gradient_evaluations"] != 14
            or row["warmup_jacobian_constructions"] != 8 or row["warmup_linear_solves"] != 8
            or row["fixture_render_evaluations"] != 4 or row["cold_forward_evaluations"] != 1
            or row["initial_diagnostic_forward_evaluations"] != 1
            or row["worker_forward_evaluations"] != warm + 6 + forwards):
        problems.append(f"coupling worker render ledger mismatch: {row['cell_id']}")
    mse = row["raw_mse"]
    if (row["psnr_ceiling_applied"] is not (mse < 1e-12)
            or (mse == 0 and row["psnr_uncapped"] is not None)
            or (mse > 0 and (row["psnr_uncapped"] is None
                            or abs(row["psnr_uncapped"] + 10 * math.log10(mse)) > 1e-6))):
        problems.append(f"coupling uncapped scoring mismatch: {row['cell_id']}")
    if row["cold_render_max_abs"] > protocol["cold_parity_max_abs"]:
        problems.append(f"coupling cold parity exceeded: {row['cell_id']}")
    initial_raw = np.load(directory / "initial_reconstruction.npy", allow_pickle=False)
    target = np.load(directory / "target.npy", allow_pickle=False)
    if initial_raw.shape != target.shape or not np.isfinite(initial_raw).all():
        problems.append(f"coupling initial image invalid: {row['cell_id']}")
    else:
        initial_mse = float(np.square(initial_raw.astype(np.float64) - target.astype(np.float64)).mean())
        if abs(-10 * math.log10(max(initial_mse, 1e-12)) - row["initial_psnr"]) > 1e-3:
            problems.append(f"coupling initial image/trace mismatch: {row['cell_id']}")
    for name in ("input_field.npz", "field.npz"):
        with np.load(directory / name, allow_pickle=False) as field:
            for key, shape in (("means", (n, 2)), ("log_scales", (n, 2)),
                               ("rotations", (n,)), ("colors", (n, 3))):
                if field[key].shape != shape or not np.isfinite(field[key]).all():
                    problems.append(f"coupling serialized field invalid: {row['cell_id']}/{name}")
            if (np.any(field["means"] < 0) or np.any(field["means"][:, 0] > protocol["shape"][1] - 1)
                    or np.any(field["means"][:, 1] > protocol["shape"][0] - 1)
                    or np.any(field["log_scales"] < math.log(protocol["config"]["scale_min"]) - 1e-6)
                    or np.any(field["log_scales"] > math.log(protocol["config"]["scale_max"]) + 1e-6)
                    or np.any(np.abs(field["colors"]) > protocol["config"]["color_limit"])):
                problems.append(f"coupling parameter bounds violated: {row['cell_id']}/{name}")


def _validate_shared_cache_scope(root, manifest, rows, problems):
    protocol = manifest["protocol"]
    if (protocol.get("timing_eligible") is not False
            or protocol.get("execution_profile") != "shared_correctness"):
        problems.append("shared cache protocol must forbid timing eligibility")
    decision = json.loads((root / "decision.json").read_text())
    if (decision.get("timing_eligible") is not False
            or decision.get("performance_disposition") != protocol["performance_disposition"]):
        problems.append("shared cache performance disposition mismatch")
    records = decision["records"]
    identities = [(r["family"], r["seed"], r["backend"]) for r in records]
    expected = {(r["family"], r["seed"], b) for r in rows for b in ("scatter", "csr")}
    if len(identities) != len(expected) or set(identities) != expected:
        problems.append("shared cache decision coverage mismatch")
    for record in records:
        if (record.get("timing_eligible") is not False or record.get("passes_speed_gate") is not False
                or record.get("passes_interchangeability_gate") != record.get("integrity_eligible")):
            problems.append("shared cache decision promotes speed or changes numerical eligibility")
    log = [json.loads(line) for line in (root / "gpu_occupancy.jsonl").read_text().splitlines()]
    if not log:
        problems.append("shared cache occupancy monitor is empty")
        return
    if log[0].get("phase") != "start" or log[-1].get("phase") != "end":
        problems.append("shared cache occupancy monitor lacks start/end")
    def valid_snapshot(value):
        return (isinstance(value, dict) and isinstance(value.get("wall_time_utc"), str)
                and isinstance(value.get("monotonic_ns"), int) and value["monotonic_ns"] > 0
                and ((value.get("status") == "ok" and isinstance(value.get("processes"), str))
                     or (value.get("status") == "error" and isinstance(value.get("error"), str))))
    if (any(not valid_snapshot(item) for item in log)
            or any(a["monotonic_ns"] > b["monotonic_ns"] for a, b in zip(log, log[1:]))):
        problems.append("shared cache occupancy monitor malformed or nonmonotonic")
        return
    for row in rows:
        if row.get("timing_eligible") is not False or row.get("execution_profile") != "shared_correctness":
            problems.append(f"shared cache row timing eligibility mismatch: {row['cell_id']}")
        if row["cell_id"] != f"{row['family']}_s{row['seed']}_r{row['repeat']}_{row['backend']}":
            problems.append(f"shared cache encoded cell identity mismatch: {row['cell_id']}")
        if row["status"] != "ok":
            continue
        directory = root / "cells" / row["cell_id"]
        config = json.loads((directory / "config.json").read_text())
        expected_solver = dict(protocol["solver"])
        expected_solver.update(basis_cache=row["backend"],
                               max_iterations=3 if manifest["diagnostic"] else protocol["solver"]["max_iterations"])
        if (config.get("timing_eligible") is not False or config.get("execution_profile") != "shared_correctness"
                or config.get("solver") != expected_solver):
            problems.append(f"shared cache config scope/solver mismatch: {row['cell_id']}")
        snapshots = []
        for label in ("before", "after"):
            path = directory / f"occupancy_{label}.json"
            snapshot = json.loads(path.read_text())
            snapshots.append(snapshot)
            if not valid_snapshot(snapshot) or sha256(path) != config[f"occupancy_{label}_sha256"]:
                problems.append(f"shared cache worker occupancy mismatch: {row['cell_id']}/{label}")
        if (snapshots[0]["monotonic_ns"] > snapshots[1]["monotonic_ns"]
                or snapshots[0]["monotonic_ns"] < log[0]["monotonic_ns"]
                or snapshots[1]["monotonic_ns"] > log[-1]["monotonic_ns"]):
            problems.append(f"shared cache occupancy interval mismatch: {row['cell_id']}")



def _validate_port_artifacts(root, rows, protocol, problems):
    """Reconstruct PORT-007 decisions from retained arrays/records, not success flags."""
    from dataclasses import asdict, replace
    from benchmarks.fit050_controls import parent_configs
    from benchmarks.port007_controls import (
        BACKENDS, coefficient_of_variation, discrete_projection, signature, summarize,
    )
    from scripts.experiments.port007_quality_reuse import decision, ellipse_mask, load_image
    from structsplat.safe_schedule import CommitTolerances
    from structsplat.pipeline import PipelineConfig, build_fit_config, build_schedule
    from structsplat.fit import _MaskConstraint
    from structsplat.safe_schedule import _quality_from_render
    import torch

    def read(path):
        return json.loads(path.read_text())

    def check(condition, message):
        if not condition:
            problems.append("PORT-007 artifact mismatch: " + message)

    def psnr(raw, target):
        mse = float(np.square(raw.astype(np.float64) - target.astype(np.float64)).mean())
        return -10 * math.log10(max(mse, 1e-12))

    def close(first, second):
        return math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-8)

    def metric_close(first, second):
        return math.isclose(first, second, rel_tol=protocol["artifact_metric_rtol"],
                            abs_tol=protocol["artifact_metric_atol"])

    def quality_from_arrays(rgb, den, target, mask, cfg, count):
        if den.shape != target.shape[:2] or rgb.shape != target.shape or mask.shape != den.shape:
            raise ValueError("PORT-007 quality array shape mismatch")
        constraint = _MaskConstraint.from_mask(mask, "cpu", torch.float32, cfg.sigma_cutoff,
            cfg.mask_margin, aa_dilation=cfg.aa_dilation, cap_mode=cfg.mask_cap_mode,
            undercoverage_band=cfg.mask_undercoverage_band)
        return _quality_from_render(torch.from_numpy(rgb), torch.from_numpy(target),
            torch.from_numpy(den), torch.from_numpy(mask), constraint, .05, count).to_dict()

    def check_quality(actual, stored, label):
        check(set(actual) == set(stored), label + " complete quality keys")
        for key, value in actual.items():
            check((stored.get(key) == value if isinstance(value, (bool, int))
                   else metric_close(stored.get(key, math.inf), value)), label + " scalar " + key)

    def field_equal(first, second):
        with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
            return set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)

    def timing(snapshots):
        return bool(snapshots) and all(
            snap.get("error") is None
            and all(p["pid"] == snap["pid"] for p in snap["compute_processes"])
            for snap in snapshots
        )

    check(read(root / "summary.json") == summarize(rows), "frozen summary")
    source = read(root / "parent_source.json")
    manifest = source["source_identity"]
    encoded = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    check(hashlib.sha256(encoded).hexdigest() == source["manifest_sha256"], "parent manifest hash")
    check(manifest["task"] == "FIT-050" and not manifest["diagnostic"]
          and manifest["repository"] == source["repository"], "parent identity")
    check(source["repository"]["commit"] == read(root / "RUNNING.json")["repository"]["commit"],
          "same-commit parent transfer")
    required_parents = {
        f"parents/coco{i:012d}_s{s}/{name}"
        for i in protocol["images"] for s in protocol["same_state"]["seeds"]
        for name in ("initial_field.npz", "field.npz", "history.json", "config.json",
                     "target.npy", "optimizer_state.pt")
    }
    check(set(source["files"]) == required_parents, "complete eight-parent transfer")
    for name, digest in source["files"].items():
        path = (root / name).resolve()
        check(path.is_relative_to(root) and path.is_file()
              and sha256(path) == digest == manifest["files"].get(name), "parent bytes " + name)
    for row in rows:
        if row.get("status") != "ok":
            continue
        label = row["cell_id"]
        directory = (root / "cells" / label).resolve()
        if not directory.is_relative_to(root):
            check(False, "escaped cell")
            continue
        config = read(directory / "config.json")
        history = read(directory / "history.json")
        stored = read(directory / "row.json")
        check(stored == row, label + " saved row")
        request = config["request"]
        check(all(request.get(k) == row.get(k) for k in
                  ("cell_id", "kind", "image", "seed", "method")), label + " request")
        with np.load(directory / "field.npz", allow_pickle=False) as field:
            check(field["means"].shape[0] == row["n_gaussians"], label + " field count")
        raw = np.load(directory / "reconstruction.npy", allow_pickle=False)
        target = np.load(directory / "target.npy", allow_pickle=False)
        check(raw.shape == target.shape and np.isfinite(raw).all()
              and np.isfinite(target).all(), label + " raw images")
        check(close(psnr(raw, target), row["psnr"]), label + " raw PSNR")
        check(metric_close(float(np.square(raw.astype(np.float64) - target.astype(np.float64)).mean()),
                           row["mse"]), label + " raw MSE")
        check(metric_close(float(np.abs(raw.astype(np.float64) - target.astype(np.float64)).mean()),
                           row["mae"]), label + " raw MAE")
        check(row["timing_eligible"] == timing(config["gpu_snapshots"]), label + " timing provenance")
        coverage, tail = BACKENDS[row["method"]]
        if row["kind"] == "same":
            _, cfg = parent_configs(row["seed"])
            expected_cfg = asdict(replace(cfg, quality_coverage_backend=coverage,
                                          quality_tail_backend=tail))
            check(config["fit"] == expected_cfg, label + " resolved fit")
            parent = root / "parents" / config["parent_id"]
            check(config["parent_config"] == read(parent / "config.json"), label + " parent config")
            check(np.array_equal(target, np.load(parent / "target.npy", allow_pickle=False)),
                  label + " paired target")
            parent_field = "initial_field.npz" if row["state"] == "initial" else "field.npz"
            check(sha256(directory / "field.npz") == sha256(parent / parent_field), label + " fixed field")
            measurements = read(directory / "measurements.json")
            parity = read(directory / "parity.json")
            prefix = f"same_coco{row['image']:012d}_s{row['seed']}_"
            base_dir = root / "cells" / (prefix + row["state"] + "_legacy_a")
            base_measurements = read(base_dir / "measurements.json")
            other = "terminal" if row["state"] == "initial" else "initial"
            other_measurements = read(root / "cells" / (prefix + other + "_legacy_a") / "measurements.json")
            checks, decisions, deltas = [], [], []
            with np.load(directory / "measurements.npz", allow_pickle=False) as data, \
                    np.load(base_dir / "measurements.npz", allow_pickle=False) as baseline:
                arrays = {key: data[key] for key in data.files}
                base_arrays = {key: baseline[key] for key in baseline.files}
                check(all(value.shape[0] == 10 for value in arrays.values()),
                      label + " ten raw measurements")
                check(np.array_equal(arrays["renders"][-1], raw), label + " selected measurement")
                for index, (record, base) in enumerate(zip(measurements, base_measurements)):
                    check(record["round"] == index, label + " round order")
                    null = decision(base["quality"], record["quality"], CommitTolerances())
                    null_control = decision(base["quality"], base["quality"], CommitTolerances())
                    changed_before = other_measurements[index]["quality"]
                    changed = decision(changed_before, record["quality"], CommitTolerances())
                    changed_control = decision(changed_before, base["quality"], CommitTolerances())
                    decisions.append({"round": index, "null": null, "null_control": null_control,
                        "changed": changed, "changed_control": changed_control,
                        "changed_direction": f"{other}_to_{row['state']}"})
                    rgb = float(np.max(np.abs(arrays["renders"][index] - base_arrays["renders"][index])))
                    replay = float(np.max(np.abs(arrays["replay_renders"][index] - base_arrays["replay_renders"][index])))
                    holes = bool(np.array_equal(arrays["hole_masks"][index], base_arrays["hole_masks"][index]))
                    check(np.array_equal(arrays["hole_masks"][index],
                                         arrays["raw_denominators"][index] < .05),
                          label + " denominator classification")
                    mask = np.ones(target.shape[:2], dtype=bool)
                    check_quality(quality_from_arrays(arrays["renders"][index],
                        arrays["raw_denominators"][index], target, mask, cfg, row["n_gaussians"]),
                        record["quality"], label + " measured")
                    check_quality(quality_from_arrays(arrays["replay_renders"][index],
                        arrays["raw_denominators"][index], target, mask, cfg, row["n_gaussians"]),
                        record["replay_quality"], label + " replay")
                    finite = bool(record["quality"]["finite"] and record["replay_quality"]["finite"]
                        and all(np.isfinite(arrays[key][index]).all() for key in
                                ("renders", "replay_renders", "raw_denominators")))
                    checks.append({"round": index, "max_rgb_error": rgb,
                        "max_replay_rgb_error": replay, "hole_mask_equal": holes,
                        "null_decision_equal": null == null_control,
                        "changed_decision_equal": changed == changed_control, "finite": finite,
                        "pass": bool(finite and rgb <= 2e-5 and replay <= 2e-5 and holes
                                     and null == null_control and changed == changed_control)})
                    deltas.append({key: record["quality"][key] - base["quality"][key]
                        for key in record["quality"]
                        if isinstance(record["quality"][key], (float, int))
                        and not isinstance(record["quality"][key], bool)})
                    check(close(psnr(arrays["renders"][index], target),
                                history["checkpoints"][index]["psnr"]), label + " temporal PSNR")
            check(parity == {"checks": checks, "decisions": decisions, "quality_deltas": deltas},
                  label + " parity recomputation")
            times = [point["seconds"] for point in measurements]
            check(len(measurements) == 10 and row["call_seconds"] == times
                  and row["total_seconds"] == sum(times)
                  and row["call_time_cv"] == coefficient_of_variation(times),
                  label + " complete call timings")
            check(row["quality"] == measurements[-1]["quality"]
                  and row["parity_pass"] == all(point["pass"] for point in checks),
                  label + " measured decision")
        else:
            expected_cfg = {**protocol["pipeline"]["config"], "seed": row["seed"],
                            "quality_coverage_backend": coverage, "quality_tail_backend": tail}
            check(config["pipeline"] == expected_cfg, label + " pipeline config")
            pipeline_cfg = PipelineConfig(**expected_cfg)
            schedule = build_schedule(pipeline_cfg)
            schedule = replace(schedule, boundary_enabled=row["image"] != 9,
                boundary=replace(schedule.boundary, name="general_closure" if row["image"] == 9
                                 else "boundary_closure"))
            fit_cfg = replace(build_fit_config(pipeline_cfg, pipeline_cfg.device),
                pixel_loss="l2", ssim_weight=0., loss_weighting="mask",
                mask_contain=schedule.boundary_enabled, mask_cap_mode="anisotropic",
                mask_undercoverage_band=float(schedule.boundary_band),
                mask_undercoverage_tau=float(schedule.coverage_tau), support_fade=True,
                coverage_match_weight=0., checkpoint_policy="terminal", triage_every=None,
                target_file_bytes=None, pool_capacity=None, split_every=None, relocate_every=None,
                prune_every=None, mask_boundary_add_every=None, adaptive_count=False, compute_lpips=False)
            check(config["fit"] == json.loads(json.dumps(asdict(fit_cfg))), label + " derived fit")
            check(config["schedule"] == json.loads(json.dumps(asdict(schedule))), label + " derived schedule")
            mask = np.load(directory / "mask.npy", allow_pickle=False)
            expected_mask = (np.ones(target.shape[:2], dtype=bool) if row["image"] == 9
                             else ellipse_mask(*target.shape[:2]))
            check(np.array_equal(mask, expected_mask)
                  and config["mask_sha256"] == sha256(directory / "mask.npy"), label + " mask")
            check(np.array_equal(target, load_image(row["image"]) * expected_mask[..., None]),
                  label + " masked input")
            reference = np.load(directory / "reference_reconstruction.npy", allow_pickle=False)
            denominator = np.load(directory / "reference_denominator.npy", allow_pickle=False)
            check(row["final_reference_rgb_max_error"] == float(np.max(np.abs(reference - raw))),
                  label + " actual replay error")
            check_quality(quality_from_arrays(reference, denominator, target, expected_mask, fit_cfg,
                                               row["n_gaussians"]), row["quality"], label + " final")
            native = history["native_events"]
            trajectory = discrete_projection(native)
            check(read(directory / "trajectory.json") == trajectory
                  and row["trajectory_sha256"] == signature(trajectory), label + " trajectory")
            check(row["iterations_run"] == history["attempted_steps"]
                  and row["accepted_steps"] == history["accepted_steps"]
                  and row["event_count"] == len(native) == len(history["checkpoints"]),
                  label + " attempted/accepted work")
            attempted = accepted = 0
            for event in native:
                attempted += event.get("attempted_steps", 0)
                accepted += event.get("accepted_steps", 0)
                check(event["global_attempted_steps"] == attempted
                      and event["global_accepted_steps"] == accepted, label + " cumulative event work")
            check(attempted == history["attempted_steps"] and accepted == history["accepted_steps"],
                  label + " actual total work")
            check(bool(native) and field_equal(directory / "field.npz", directory / "snapshots" /
                                               f"field_{len(native) - 1:04d}.npz"), label + " final selected field")
            selected = max((event["global_attempted_steps"] for event in native
                            if event["accepted"]), default=0)
            check(row["selected_iteration"] == selected, label + " selected step")
            check(row["observer_seconds"] == config["observer_seconds"]
                  and row["total_seconds"] == config["instrumented_total_seconds"],
                  label + " inclusive timing")
            for index, (event, point) in enumerate(zip(native, history["checkpoints"])):
                check(point["attempted_steps"] == event["global_attempted_steps"]
                      and all(point[k] == v for k, v in event["selected"].items()),
                      label + " native checkpoint")
                with np.load(directory / "snapshots" / f"field_{index:04d}.npz", allow_pickle=False) as field:
                    check(field["means"].shape[0] == event["selected"]["n_gaussians"],
                          label + " snapshot count")

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
        if manifest["protocol"].get("execution_profile") == "shared_correctness":
            _validate_shared_cache_scope(root, manifest, rows, problems)
        if manifest["task"] in {"FIT-050", "PORT-007", "FIT-051"}:
            from importlib import import_module
            module = import_module({
                "FIT-050": "scripts.experiments.fit050_color_ray",
                "FIT-051": "scripts.experiments.fit051_actual_color_ray",
                "PORT-007": "benchmarks.port007_controls",
            }[manifest["task"]])
            if manifest["task"] in {"FIT-050", "FIT-051"}:
                module.validate_rows(rows, manifest["protocol"], problems,
                                     diagnostic=manifest["diagnostic"])
                module.validate_artifacts(root, rows, manifest["protocol"], problems,
                                          diagnostic=manifest["diagnostic"])
                declared = module.expected_cells(manifest["protocol"], manifest["diagnostic"])
            else:
                module.validate_rows(rows, manifest["protocol"], problems)
                _validate_port_artifacts(root, rows, manifest["protocol"], problems)
                declared = module.expected_cells()
            if sorted(declared) != sorted(expected):
                problems.append("declared matrix disagrees with task executable protocol")
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
                elif manifest["task"] == "HIER-036":
                    _validate_coupling_cell(root, row, manifest["protocol"], manifest["diagnostic"], problems)
                elif manifest["task"] == "HIER-033":
                    _validate_oracle_cell(root, row, manifest["protocol"], manifest["diagnostic"], problems)
            for name in row.get("artifacts", {}).values():
                if name not in files or (root / name).resolve() not in linked:
                    problems.append(f"unexposed or unhashed artifact: {name}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(f"malformed HIER research bundle: {exc}")
    return problems
