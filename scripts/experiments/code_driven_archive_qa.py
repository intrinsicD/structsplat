#!/usr/bin/env python3
"""Read-only QA of the three-study partial archive; optionally preserve browser receipts.

The optional browser directory is new post-run packaging, never an experiment or a changed
archive. No screenshot is transformed and no browser profile or original bundle is copied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import unquote, urlsplit


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), f'Uncontained path: {name}')
    return path


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.rows, self.cells = [], [], []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ('href', 'src') and value)
        if tag == 'tr':
            self.cells = []
        elif tag == 'td':
            self.cell = ''

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag == 'td':
            self.cells.append(self.cell)
            self.cell = None
        elif tag == 'tr' and self.cells:
            self.rows.append(self.cells)


def audit(archive, repository):
    archive, repository = Path(archive).resolve(), Path(repository).resolve()
    inventory = json.loads((archive / 'archive_manifest.json').read_text())
    require(inventory['profile'] == 'code-driven' and not inventory['complete_replay_bundle'], 'Wrong archive scope')
    require(set(inventory['bundles']) == {'fit050', 'port007', 'fit051'}, 'Wrong study inventory')
    require(sha(archive / 'packager.py') == inventory['packager_source_sha256'], 'Archive packager hash differs')
    report = {'archive_manifest_sha256': sha(archive / 'archive_manifest.json'), 'bundles': {},
              'total_rows': 0, 'copied_files_checked': 0, 'generated_files_checked': 0,
              'local_html_links_checked': 0, 'html_pages_checked': 0}
    expected_payloads = {'archive_manifest.json'}
    for label, entry in inventory['bundles'].items():
        destination = archive / label
        original = Path(entry['original_bundle']).resolve()
        manifest_path = destination / 'original_manifest.json'
        require(sha(manifest_path) == entry['original_manifest_sha256'] == sha(original / 'manifest.json'),
                f'{label}: original manifest binding differs')
        manifest = json.loads(manifest_path.read_text())
        require(not manifest['diagnostic'] and not manifest['repository']['dirty'], f'{label}: source not formal/clean')
        require(entry['source_commit'] == manifest['repository']['commit']
                and entry['protocol_digest'] == manifest['protocol_digest'], f'{label}: source/protocol identity differs')
        encoded = json.dumps({'protocol': manifest['protocol'], 'source': manifest['source_files']},
                             sort_keys=True, separators=(',', ':')).encode()
        require(hashlib.sha256(encoded).hexdigest() == manifest['protocol_digest'], f'{label}: protocol digest differs')
        for name, expected in manifest['source_files'].items():
            blob = subprocess.check_output(['git', 'cat-file', 'blob', f'{entry["source_commit"]}:{name}'],
                                           cwd=repository, timeout=30)
            require(hashlib.sha256(blob).hexdigest() == expected, f'{label}: committed source differs: {name}')
        # Deliberately independent from the archive writer's selection function.
        selected = {name for name in manifest['files']
                    if Path(name).suffix in ('.json', '.jsonl')
                    or (len(Path(name).parts) == 1 and Path(name).suffix == '.csv')
                    or Path(name).name in ('field.npz', 'initial_field.npz', 'candidate_field.npz')}
        require(set(entry['copied_files']) == selected | {'original_manifest.json'}, f'{label}: exact selection differs')
        require(entry['omitted_files'] == sorted(set(manifest['files']) - selected), f'{label}: exact omissions differ')
        for name, expected in entry['copied_files'].items():
            target = inside(destination, name)
            source = original / 'manifest.json' if name == 'original_manifest.json' else inside(original, name)
            require(sha(target) == expected == sha(source), f'{label}: copied/source hash differs: {name}')
            if name != 'original_manifest.json':
                require(expected == manifest['files'][name], f'{label}: original artifact binding differs: {name}')
            expected_payloads.add(f'{label}/{name}')
        rows = json.loads((destination / 'metrics.json').read_text())
        ids = [row['cell_id'] for row in rows]
        require(len(ids) == len(set(ids)) == entry['expected_cells']
                and sorted(ids) == sorted(manifest['expected_cells']), f'{label}: cell matrix differs')
        jsonl = [json.loads(line) for line in (destination / 'metrics.jsonl').read_text().splitlines() if line]
        require(jsonl == rows, f'{label}: JSONL metric rows differ')
        with (destination / 'metrics.csv').open(newline='') as stream:
            csv_rows = list(csv.DictReader(stream))
        require([row['cell_id'] for row in csv_rows] == ids, f'{label}: CSV cell rows differ')
        page = Page()
        page.feed((destination / 'index.html').read_text())
        require([row[0] for row in page.rows] == ids, f'{label}: HTML metric rows differ')
        report['bundles'][label] = {'rows': len(rows), 'copied_files': len(entry['copied_files']),
            'omitted_files': len(entry['omitted_files']), 'source_files_checked': len(manifest['source_files']),
            'source_commit': entry['source_commit'], 'protocol_digest': entry['protocol_digest'],
            'original_manifest_sha256': entry['original_manifest_sha256'], 'original_bundle': str(original)}
        report['total_rows'] += len(rows)
        report['copied_files_checked'] += len(entry['copied_files'])
    for name, expected in inventory['generated_files'].items():
        require(sha(inside(archive, name)) == expected, f'Generated artifact hash differs: {name}')
        expected_payloads.add(name)
        report['generated_files_checked'] += 1
    require({str(path.relative_to(archive)) for path in archive.rglob('*') if path.is_file()} == expected_payloads,
            'Archive contains missing or undeclared files')
    for path in archive.rglob('*.html'):
        page = Page()
        page.feed(path.read_text())
        for link in page.links:
            parsed = urlsplit(link)
            require(not parsed.scheme and not parsed.netloc, f'Nonportable HTML link: {path.name}: {link}')
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            require(target.is_relative_to(archive) and target.is_file(), f'Broken/escaping HTML link: {link}')
            report['local_html_links_checked'] += 1
        report['html_pages_checked'] += 1
    require(report['total_rows'] == 214, 'Three-study archive must preserve all 214 rows')
    report['status'] = 'passed'
    return report


def preserve_browser(source, output, archive, qa):
    source, output, archive = Path(source).resolve(), Path(output).resolve(), Path(archive).resolve()
    require(not output.is_relative_to(source) and not output.is_relative_to(archive), 'Unsafe browser output location')
    names = [f'{label}-v1-{suffix}' for label in ('fit050', 'port007', 'fit051')
             for suffix in ('browser.json', 'overview.png', 'native.png')]
    curves = sorted(path.name for path in source.glob('fit051-v1-curve*.png') if path.is_file())
    names += curves
    if (source / 'review.mjs').is_file():
        names.append('review.mjs')
    hashes = {name: sha(inside(source, name)) for name in names}
    receipts = {}
    for label, study in qa['bundles'].items():
        original = Path(study['original_bundle'])
        require(not output.is_relative_to(original), 'Browser output would alter an original bundle')
        receipt = json.loads((source / f'{label}-v1-browser.json').read_text())
        require(Path(receipt['report']).resolve() == original / 'index.html', f'{label}: receipt source differs')
        require(receipt['overview']['rows'] == study['rows'] and not receipt['overview']['broken'],
                f'{label}: receipt matrix or prior broken-image checks differ')
        receipts[label] = {'original_report_sha256': sha(original / 'index.html'),
                           'original_manifest_sha256': study['original_manifest_sha256'],
                           'source_commit': study['source_commit'], 'protocol_digest': study['protocol_digest']}
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in hashes.items():
        shutil.copyfile(source / name, output / name)
        require(sha(output / name) == expected == sha(source / name), f'Browser evidence changed during copy: {name}')
    shutil.copyfile(__file__, output / 'packager.py')
    inventory = {'schema': 'structsplat.code-driven.browser-evidence.v1', 'source_directory': str(source),
        'copied_files': hashes, 'packager_source_sha256': sha(__file__),
        'generated_files': {'packager.py': sha(output / 'packager.py')},
        'optional_fit051_curve_screenshots': curves, 'receipt_bindings': receipts, 'archive_qa': qa,
        'scope': 'Byte-preserved prior browser JSON receipts and screenshots, not a new browser review or image edit. '
                 'Chrome profile/cache omitted. No scientific conclusions are added; original report bundles remain authoritative.',
        'reproduce': ['python', str(Path(__file__).resolve()), str(archive), '--repository',
                      str(Path(__file__).resolve().parents[2]), '--browser-source', str(source), '--browser-out', 'NEW_DIRECTORY']}
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')
    return inventory


def refresh_browser(output, archive, qa):
    """Append archive QA history without replacing any screenshot or earlier receipt bytes."""
    output, archive = Path(output).resolve(), Path(archive).resolve()
    require(not output.is_relative_to(archive), 'Browser inventory cannot be inside the archive')
    path = output / 'inventory.json'
    previous_bytes = path.read_bytes()
    previous_hash = hashlib.sha256(previous_bytes).hexdigest()
    inventory = json.loads(previous_bytes)
    for name, expected in {**inventory['copied_files'], **inventory['generated_files']}.items():
        require(sha(inside(output, name)) == expected, f'Existing browser evidence hash differs: {name}')
    previous_name = f'inventory-{previous_hash[:12]}.json'
    require(not (output / previous_name).exists(), 'Previous inventory receipt already preserved')
    with (output / previous_name).open('xb') as stream:
        stream.write(previous_bytes)
    script_hash = sha(__file__)
    script_name = f'qa-packager-{script_hash[:12]}.py'
    if (output / script_name).exists():
        require(sha(output / script_name) == script_hash, 'Preserved QA program hash differs')
    else:
        shutil.copyfile(__file__, output / script_name)
    history = inventory.setdefault('archive_qa_history', [inventory['archive_qa']])
    history.append(qa)
    inventory['archive_qa'] = qa
    inventory['previous_inventory'] = {'path': previous_name, 'sha256': previous_hash}
    inventory['generated_files'].update({previous_name: previous_hash, script_name: script_hash})
    inventory['latest_qa_script_sha256'] = script_hash
    inventory['refresh_reproduce'] = ['python', str(Path(__file__).resolve()), str(archive),
                                      '--browser-out', str(output), '--refresh-browser-inventory']
    path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--browser-source', type=Path)
    parser.add_argument('--browser-out', type=Path)
    parser.add_argument('--refresh-browser-inventory', action='store_true')
    args = parser.parse_args()
    if args.refresh_browser_inventory:
        if args.browser_out is None or args.browser_source is not None:
            parser.error('Refresh requires --browser-out and no --browser-source')
    elif (args.browser_source is None) != (args.browser_out is None):
        parser.error('--browser-source and --browser-out must be supplied together')
    qa = audit(args.archive, args.repository)
    if args.refresh_browser_inventory:
        refresh_browser(args.browser_out, args.archive, qa)
    elif args.browser_source is not None:
        preserve_browser(args.browser_source, args.browser_out, args.archive, qa)
    print(json.dumps(qa, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
