"""Portable fault checks for the explicitly partial, post-run evidence package."""
import json

import pytest

from scripts.experiments.overnight_evidence_archive import archive, contained, digest, selection


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
