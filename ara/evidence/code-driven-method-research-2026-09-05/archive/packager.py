#!/usr/bin/env python3
"""Package an explicitly partial, hash-checked archive of the September 5 assays.

This is post-run evidence packaging, not an experiment or a new decision rule.
The original immutable bundles remain the complete replay/report authority.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Uncontained artifact: {relative}')
    return path


def inspect_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['diagnostic'] or manifest['repository']['dirty']:
        raise ValueError('Only completed clean-source formal bundles may be archived')
    if 'COMPLETED.json' not in manifest['files']:
        raise ValueError('Incomplete bundle')
    for relative, expected in manifest['files'].items():
        if digest(contained(root, relative)) != expected:
            raise ValueError(f'Hash mismatch: {root.name}/{relative}')
    rows = json.loads((root / 'metrics.json').read_text())
    if sorted(row['cell_id'] for row in rows) != sorted(manifest['expected_cells']):
        raise ValueError('Cell matrix mismatch')
    return manifest, rows


def selection(relative, profile='overnight'):
    """All scalar/decision/field evidence; display examples use seed0, first repeat."""
    path = Path(relative)
    if profile == 'code-driven':
        return (path.suffix in {'.json', '.jsonl'}
                or (len(path.parts) == 1 and path.suffix == '.csv')
                or path.name in {'field.npz', 'initial_field.npz', 'candidate_field.npz'})
    if profile != 'overnight':
        raise ValueError(f'Unknown archive profile: {profile}')
    if len(path.parts) == 1:
        return path.suffix in {'.json', '.csv', '.jsonl'}
    if path.name in {'config.json', 'request.json', 'row.json', 'history.json',
                     'field.npz', 'input_field.npz', 'base_field.npz',
                     'gradient_packet.npz', 'occupancy_before.json', 'occupancy_after.json'}:
        return True
    if path.suffix == '.png' and '_s0_' in relative:
        repeat = re.search(r'_r(\d+)_', relative)
        return repeat is None or repeat.group(1) == '0'
    return False


def archive(outdir, bundles, *, timing_sidecar=None, profile='overnight'):
    if profile not in {'overnight', 'code-driven'}:
        raise ValueError(f'Unknown archive profile: {profile}')
    bundles = list(bundles)
    if not bundles:
        raise ValueError('At least one bundle is required')
    outdir = Path(outdir).resolve()
    roots = [Path(root).resolve() for _, root in bundles]
    if timing_sidecar is not None:
        roots.append(Path(timing_sidecar).resolve())
    if any(outdir.is_relative_to(root) for root in roots):
        raise ValueError('Archive must not be inside an immutable input bundle')
    inspected = [(label, Path(root).resolve(), *inspect_bundle(root)) for label, root in bundles]
    if len({label for label, *_ in inspected}) != len(inspected):
        raise ValueError('Duplicate bundle labels')
    for label, *_ in inspected:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
            raise ValueError('Unsafe label')
    outdir.mkdir(parents=True, exist_ok=False)
    code_driven = profile == 'code-driven'
    scope = ('All root JSON/JSONL/CSV and all nested JSON/JSONL; selected and parent native files '
             'named field.npz, initial_field.npz and candidate_field.npz. Input-field duplicates, '
             'per-event field_*.npz snapshots, optimizer states, raw float rasters, trial arrays, PNG displays '
             'and original HTML are omitted. The original manifest describes the full bundle, '
             'not this partial archive.' if code_driven else
             'All decision/scalar/configuration/history/native-field evidence; selected seed0 display examples. '
             'Raw float rasters, duplicate progress logs, most display assets and original HTML are omitted. '
             'The original manifest describes the full bundle, not this partial archive.')
    inventory = {'schema': f'structsplat.{profile}.partial-archive.v1', 'complete_replay_bundle': False,
                 'packager_source_sha256': digest(__file__),
                 'selection': scope if code_driven else selection.__doc__, 'bundles': {}}
    if code_driven:
        inventory['profile'] = profile
    title = 'StructSplat code-driven evidence' if code_driven else 'StructSplat overnight evidence'
    sections = []
    if timing_sidecar is not None:
        sidecar = Path(timing_sidecar).resolve()
        metadata = json.loads((sidecar / 'metadata.json').read_text())
        if digest(sidecar / 'monitor.py') != metadata['monitor_sha256']:
            raise ValueError('Timing monitor source mismatch')
        completion = json.loads((sidecar / 'completion.json').read_text())
        if completion['returncode'] != 0:
            raise ValueError('Timing sidecar did not record successful driver completion')
        samples = [json.loads(line) for line in (sidecar / 'occupancy.jsonl').read_text().splitlines()]
        if not samples or samples[0]['phase'] != 'preflight' or samples[-1]['phase'] != 'end':
            raise ValueError('Incomplete timing occupancy log')
        target = outdir / 'timing_sidecar'
        target.mkdir()
        hashes = {}
        for name in ('metadata.json', 'completion.json', 'monitor.py', 'occupancy.jsonl'):
            shutil.copyfile(sidecar / name, target / name)
            hashes[name] = digest(target / name)
        inventory['timing_sidecar'] = {
            'original_path': str(sidecar), 'copied_files': hashes,
            'samples': len(samples),
            'foreign_process_samples': sum(any(p['owned_by_driver'] is False for p in row.get('processes', [])) for row in samples),
            'unclassified_process_samples': sum(any(p['owned_by_driver'] is None for p in row.get('processes', [])) for row in samples),
            'query_errors': sum(row['status'] != 'ok' for row in samples),
            'qualification': 'Point samples do not establish continuous GPU or workstation exclusivity. '
                             'Any foreign activity prevents an unqualified isolated-speed interpretation. '
                             + ('' if code_driven else
                                'CPU contention was separately observed in the research audit.')}
    for label, root, manifest, rows in inspected:
        destination = outdir / label
        destination.mkdir()
        shutil.copyfile(root / 'manifest.json', destination / 'original_manifest.json')
        copied = {'original_manifest.json': digest(destination / 'original_manifest.json')}
        for relative, expected in manifest['files'].items():
            if not selection(relative, profile):
                continue
            target = contained(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(contained(root, relative), target)
            if digest(target) != expected:
                raise ValueError('Source changed during archive')
            copied[relative] = expected
        inventory['bundles'][label] = {
            'original_bundle': str(root), 'original_manifest_sha256': digest(root / 'manifest.json'),
            'source_commit': manifest['repository']['commit'], 'protocol_digest': manifest['protocol_digest'],
            'expected_cells': len(manifest['expected_cells']), 'copied_files': copied,
            'omitted_files': sorted(set(manifest['files']) - set(copied)),
            'scope': scope}
        columns = [key for key in ('cell_id', 'status', 'psnr', 'ms_ssim', 'lpips', 'selected_iteration',
                   'forward_evaluations', 'forward_calls', 'rejected_updates') if any(key in row for row in rows)]
        table = '<table><thead><tr>' + ''.join(f'<th>{html.escape(key)}</th>' for key in columns) + '</tr></thead><tbody>'
        for row in rows:
            table += '<tr>' + ''.join(f'<td>{html.escape(str(row.get(key, "")))}</td>' for key in columns) + '</tr>'
        table += '</tbody></table>'
        figures = ''.join(f'<figure><a href="{html.escape(relative)}"><img loading="lazy" src="{html.escape(relative)}"></a>'
                          f'<figcaption>{html.escape(relative)}</figcaption></figure>'
                          for relative in copied if relative.endswith('.png'))
        page = f'<h1>{html.escape(label)}: partial evidence archive</h1><p>{html.escape(inventory["bundles"][label]["scope"])}</p>'
        page += '<p><a href="../index.html">Archive index</a> · '
        decision = next((name for name in ('decision.json', 'summary.json') if name in copied), None)
        if decision is not None:
            page += f'<a href="{decision}">Frozen decisions</a> · '
        page += '<a href="metrics.csv">All metrics CSV</a> · <a href="metrics.json">All metrics JSON</a> · '
        page += '<a href="original_manifest.json">Original full-bundle manifest</a></p>'
        displays = '' if code_driven else '<h2>Illustrative displays, not a selection gate</h2>' + figures
        (destination / 'index.html').write_text(document(page + table + displays, title=title))
        sections.append(f'<li><a href="{label}/index.html">{html.escape(label)}</a>: {len(rows)} cells; '
                        f'source <code>{manifest["repository"]["commit"]}</code></li>')
    if code_driven:
        intro = '<h1>September 5 code-driven research: evidence archive</h1><p>This is an explicitly partial '
        intro += f'post-run archive of {len(inspected)} code-driven studies, not new experiments or a complete replay bundle. '
        intro += 'No filtering of metric rows or decisions. The original immutable result directories retain '
        intro += 'every raw artifact and full native report. Component-only observations do not establish '
        intro += 'a default-method improvement, an isolated end-to-end pipeline speedup, held-out generalization '
        intro += 'or research novelty. No default, isolated-pipeline, held-out or novelty claim is made by this archive. '
        intro += 'See each original study decision and independent audit for its measured scope and limitations.'
    else:
        intro = '<h1>September 5 overnight research: evidence archive</h1><p>This is an explicitly partial '
        intro += 'archive, not five new experiments or a complete replay bundle. No filtering of metric rows or decisions. '
        intro += 'The original immutable result directories retain every raw artifact and full native report. '
        intro += 'Shared-profile cache timing is ineligible; optimizer timing is descriptive. Original cache timing '
        intro += 'is observed shared-workstation timing: external GPU activity and CPU compilation were observed. '
        intro += 'Nominal frozen speed gates are not isolated-speed evidence; passing rollback-to-input workloads '
        intro += 'do not establish acceleration of an accepted fitted-coefficient update. See the research handoff '
        intro += 'and independent audit for interpretation and limitations.'
    intro += '</p><p><a href="archive_manifest.json">'
    intro += 'Input hashes, archive inventory and exact omissions</a></p><ul>' + ''.join(sections) + '</ul>'
    if timing_sidecar is not None:
        intro += '<p><a href="timing_sidecar/occupancy.jsonl">Original timing occupancy log</a> · '
        intro += '<a href="timing_sidecar/metadata.json">Monitor identity and command</a></p>'
    (outdir / 'index.html').write_text(document(intro, title=title))
    shutil.copyfile(__file__, outdir / 'packager.py')
    inventory['generated_files'] = {str(path.relative_to(outdir)): digest(path)
                                    for path in sorted(outdir.rglob('index.html'))}
    inventory['generated_files']['packager.py'] = digest(outdir / 'packager.py')
    (outdir / 'archive_manifest.json').write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')
    return inventory


def document(body, *, title='StructSplat overnight evidence'):
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1"><title>' + html.escape(title) + '</title>'
            '<style>body{font:15px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#17212d}'
            'table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #ccd3da;padding:5px;text-align:left}'
            'th{background:#edf2f6}figure{display:inline-block;vertical-align:top;margin:12px;max-width:620px}'
            'img{max-width:100%;height:auto}figcaption{overflow-wrap:anywhere}code{overflow-wrap:anywhere}</style>'
            + body + '</html>')


def bundle_argument(value):
    label, separator, path = value.partition('=')
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError('Bundle must be a nonempty LABEL=PATH assignment')
    return label, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outdir')
    parser.add_argument('--bundle', action='append', required=True, type=bundle_argument, help='LABEL=PATH')
    parser.add_argument('--timing-sidecar', help='Completed external occupancy sidecar directory')
    parser.add_argument('--profile', choices=('overnight', 'code-driven'), default='overnight')
    args = parser.parse_args()
    archive(args.outdir, args.bundle, timing_sidecar=args.timing_sidecar, profile=args.profile)


if __name__ == '__main__':
    main()
