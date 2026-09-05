"""Portable fault checks for the explicitly partial, post-run evidence package."""
import json

import pytest

from scripts.experiments.overnight_evidence_archive import archive, contained, digest, main, selection


def fixture_bundle(root):
    root.mkdir()
    (root / 'metrics.json').write_text(json.dumps([{'cell_id': 'one', 'status': 'ok', 'psnr': 20}]))
    (root / 'COMPLETED.json').write_text('{}')
    (root / 'decision.json').write_text('{"default_promotion":false}')
    (root / 'metrics.csv').write_text('cell_id,status,psnr\none,ok,20\n')
    files = {p.name: digest(p) for p in root.iterdir()}
    (root / 'manifest.json').write_text(json.dumps({'diagnostic': False, 'repository': {'dirty': False, 'commit': 'abc'},
        'files': files, 'expected_cells': ['one'], 'protocol_digest': 'def'}))
    return root


def test_archive_is_explicitly_partial_and_preserves_decisions(tmp_path):
    source = fixture_bundle(tmp_path / 'source')
    output = tmp_path / 'archive'
    result = archive(output, [('assay', source)])
    assert result['complete_replay_bundle'] is False
    assert (output / 'assay/decision.json').read_bytes() == (source / 'decision.json').read_bytes()
    assert result['bundles']['assay']['original_manifest_sha256'] == digest(source / 'manifest.json')
    assert not (output / 'assay/manifest.json').exists()
    with pytest.raises(FileExistsError):
        archive(output, [('assay', source)])


def test_hash_failure_creates_no_archive(tmp_path):
    source = fixture_bundle(tmp_path / 'source')
    (source / 'decision.json').write_text('{}')
    with pytest.raises(ValueError, match='Hash mismatch'):
        archive(tmp_path / 'archive', [('assay', source)])
    assert not (tmp_path / 'archive').exists()


def test_nested_output_cannot_modify_immutable_input(tmp_path):
    source = fixture_bundle(tmp_path / 'source')
    before = sorted(p.name for p in source.iterdir())
    with pytest.raises(ValueError, match='immutable input'):
        archive(source / 'archive', [('assay', source)])
    assert sorted(p.name for p in source.iterdir()) == before


def test_external_timing_occupancy_is_preserved_and_qualified(tmp_path):
    source = fixture_bundle(tmp_path / 'source')
    sidecar = tmp_path / 'sidecar'
    sidecar.mkdir()
    (sidecar / 'monitor.py').write_text('# observational monitor\n')
    (sidecar / 'metadata.json').write_text(json.dumps({'monitor_sha256': digest(sidecar / 'monitor.py')}))
    (sidecar / 'completion.json').write_text('{"returncode":0}')
    samples = [{'phase': 'preflight', 'status': 'ok', 'processes': []},
               {'phase': 'sample', 'status': 'ok', 'processes': [{'owned_by_driver': False}]},
               {'phase': 'end', 'status': 'ok', 'processes': []}]
    (sidecar / 'occupancy.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in samples))
    output = tmp_path / 'archive'
    result = archive(output, [('assay', source)], timing_sidecar=sidecar)
    assert result['timing_sidecar']['foreign_process_samples'] == 1
    assert (output / 'timing_sidecar/occupancy.jsonl').read_bytes() == (sidecar / 'occupancy.jsonl').read_bytes()
    assert 'not isolated-speed evidence' in (output / 'index.html').read_text()


def test_selection_and_containment(tmp_path):
    assert selection('cells/texture_s5_full_shared/history.json')
    assert selection('cells/texture_s0_full_shared/curves.png')
    assert selection('cells/texture_s0_full_row/curves.png')
    assert not selection('cells/texture_s5_full_shared/curves.png')
    assert not selection('cells/hier031_s0_r0_off/reconstruction.npy')
    assert selection('cells/hier031_s0_r0_off/reconstruction.png')
    assert not selection('cells/hier031_s0_r1_off/reconstruction.png')
    with pytest.raises(ValueError, match='Uncontained'):
        contained(tmp_path, '../escape')


@pytest.mark.parametrize('relative,expected', [
    ('metrics.json', True), ('metrics.jsonl', True), ('metrics.csv', True),
    ('parents/p/config.json', True), ('parents/p/occupancy.jsonl', True),
    ('cells/p/tensor_inventory.json', True), ('cells/p/measurements.json', True),
    ('cells/p/nested/unusual.jsonl', True), ('cells/p/measurements.csv', False),
    ('parents/p/field.npz', True), ('parents/p/initial_field.npz', True),
    ('cells/p/candidate_field.npz', True), ('cells/p/field.npz', True),
    ('cells/p/input_field.npz', False), ('cells/p/base_field.npz', False),
    ('cells/p/snapshotfield_1.npz', False), ('cells/p/field_1.npz', False),
    ('parents/p/optimizer.pt', False), ('parents/p/optimizer_state.pt', False),
    ('cells/p/transaction_arrays.npz', False), ('cells/p/trial_arrays.npz', False),
    ('cells/p/reconstruction.npy', False), ('cells/p_s0_/reconstruction.png', False),
    ('cells/p_s0_/curves.png', False), ('index.html', False),
])
def test_code_driven_selection_preserves_metadata_and_only_named_native_fields(relative, expected):
    assert selection(relative, 'code-driven') is expected


def enrich_code_bundle(root, *, summary=False):
    rows = [{'cell_id': 'one', 'status': 'ok', 'psnr': 20},
            {'cell_id': 'two', 'status': 'error', 'error': 'retained failure'}]
    extras = {'metrics.json': json.dumps(rows),
        'metrics.jsonl': ''.join(json.dumps(row) + '\n' for row in rows),
        'metrics.csv': 'cell_id,status,psnr\none,ok,20\ntwo,error,\n',
        'parents/p/field.npz': 'native parent', 'parents/p/initial_field.npz': 'native initial',
        'parents/p/optimizer_state.pt': 'omitted moments',
        'cells/p/field.npz': 'native selected', 'cells/p/candidate_field.npz': 'native proposal',
        'cells/p/input_field.npz': 'duplicate parent', 'cells/p/snapshotfield_1.npz': 'snapshot',
        'cells/p/transaction_arrays.npz': 'trial arrays', 'cells/p/reconstruction.npy': 'raw raster',
        'cells/p_s0_/reconstruction.png': 'display', 'cells/p/measurements.csv': 'nested csv',
        'cells/p/tensor_inventory.json': '{}', 'cells/p/deep/decisions.jsonl': '{}\n'}
    for relative, value in extras.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    if summary:
        (root / 'decision.json').rename(root / 'summary.json')
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['expected_cells'] = [row['cell_id'] for row in rows]
    manifest['files'] = {str(path.relative_to(root)): digest(path)
                         for path in root.rglob('*') if path.is_file() and path != root / 'manifest.json'}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


def test_code_driven_archive_has_own_narrative_exact_omissions_and_all_rows(tmp_path):
    bundles = [(label, enrich_code_bundle(fixture_bundle(tmp_path / label), summary=label == 'port007'))
               for label in ('fit050', 'port007', 'fit051')]
    output = tmp_path / 'archive'
    result = archive(output, bundles, profile='code-driven')
    assert result['profile'] == 'code-driven'
    assert result['schema'] == 'structsplat.code-driven.partial-archive.v1'
    assert not result['complete_replay_bundle']
    index = (output / 'index.html').read_text()
    for text in ('3 code-driven studies', 'Component-only observations',
                 'No default, isolated-pipeline, held-out or novelty claim', 'not new experiments'):
        assert text in index
    for text in ('five new experiments', 'Shared-profile cache', 'CPU compilation were observed',
                 'external GPU activity', 'rollback-to-input workloads', 'overnight evidence'):
        assert text not in index
    for label, source in bundles:
        info = result['bundles'][label]
        source_manifest = json.loads((source / 'manifest.json').read_text())
        expected = {name for name in source_manifest['files'] if selection(name, 'code-driven')}
        assert set(info['copied_files']) == expected | {'original_manifest.json'}
        assert set(info['omitted_files']) == set(source_manifest['files']) - expected
        assert info['expected_cells'] == 2
        assert all(digest(output / label / name) == value for name, value in info['copied_files'].items())
        for name in ('metrics.json', 'metrics.jsonl', 'metrics.csv'):
            assert (output / label / name).read_bytes() == (source / name).read_bytes()
        page = (output / label / 'index.html').read_text()
        assert '<td>one</td>' in page and '<td>two</td>' in page and '<td>error</td>' in page
        assert 'Illustrative displays' not in page
        assert 'optimizer states' in page and 'not this partial archive' in page
        decision = 'summary.json' if label == 'port007' else 'decision.json'
        assert f'href="{decision}"' in page
        assert (output / label / decision).read_bytes() == (source / decision).read_bytes()
    assert 'href="decision.json"' not in (output / 'port007/index.html').read_text()


def test_code_driven_cli_profile_and_malformed_input_guards(tmp_path, monkeypatch):
    source = fixture_bundle(tmp_path / 'source')
    output = tmp_path / 'archive'
    monkeypatch.setattr('sys.argv', ['overnight_evidence_archive.py', str(output),
                                    '--profile', 'code-driven', '--bundle', f'fit051={source}'])
    main()
    assert json.loads((output / 'archive_manifest.json').read_text())['profile'] == 'code-driven'
    for assignment in ('missing_equals', '=path', 'label='):
        invalid = tmp_path / 'invalid'
        monkeypatch.setattr('sys.argv', ['overnight_evidence_archive.py', str(invalid), '--bundle', assignment])
        with pytest.raises(SystemExit) as raised:
            main()
        assert raised.value.code == 2 and not invalid.exists()
    with pytest.raises(ValueError, match='Unknown archive profile'):
        archive(tmp_path / 'unknown', [('fit051', source)], profile='unknown')
    assert not (tmp_path / 'unknown').exists()
    with pytest.raises(ValueError, match='At least one bundle'):
        archive(tmp_path / 'empty', [], profile='code-driven')
    assert not (tmp_path / 'empty').exists()


def test_code_driven_profile_does_not_weaken_source_hash_checks(tmp_path):
    source = enrich_code_bundle(fixture_bundle(tmp_path / 'source'))
    (source / 'parents/p/optimizer_state.pt').write_text('tampered omitted artifact')
    with pytest.raises(ValueError, match='Hash mismatch'):
        archive(tmp_path / 'archive', [('fit051', source)], profile='code-driven')
    assert not (tmp_path / 'archive').exists()
