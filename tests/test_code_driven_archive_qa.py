"""Portable packaging checks; no experimental images or numerical work."""
import copy
import json

import pytest

from scripts.experiments.code_driven_archive_qa import Page, inside, preserve_browser, refresh_browser, sha


def receipts(tmp_path):
    source, archive = tmp_path / 'source', tmp_path / 'archive'
    source.mkdir()
    archive.mkdir()
    qa = {'archive_manifest_sha256': 'a' * 64, 'bundles': {}}
    for label in ('fit050', 'port007', 'fit051'):
        original = tmp_path / label
        original.mkdir()
        (original / 'index.html').write_text('<html>Previously reviewed report</html>')
        receipt = {'report': str(original / 'index.html'), 'overview': {'rows': 1, 'broken': []}}
        (source / f'{label}-v1-browser.json').write_text(json.dumps(receipt))
        for suffix in ('overview.png', 'native.png'):
            (source / f'{label}-v1-{suffix}').write_bytes(b'byte-preserved screenshot fixture')
        qa['bundles'][label] = {'original_bundle': str(original), 'rows': 1,
            'original_manifest_sha256': label, 'source_commit': 'commit', 'protocol_digest': 'protocol'}
    (source / 'fit051-v1-curves.png').write_bytes(b'optional curve fixture')
    return source, archive, qa


def test_html_qa_collects_every_local_reference_and_metric_row():
    page = Page()
    page.feed('<a href="summary.json">Decision</a><img src="image.png"><table><tr><th>ID</th></tr>'
              '<tr><td>cell1</td><td>error</td></tr><tr><td>cell2</td><td>ok</td></tr></table>')
    assert page.links == ['summary.json', 'image.png']
    assert page.rows == [['cell1', 'error'], ['cell2', 'ok']]


def test_qa_containment_rejects_escape(tmp_path):
    with pytest.raises(ValueError, match='Uncontained'):
        inside(tmp_path, '../escape')


def test_browser_refresh_preserves_old_receipt_screenshot_bytes_and_both_qa_bindings(tmp_path):
    source, archive, qa = receipts(tmp_path)
    output = tmp_path / 'browser'
    first = preserve_browser(source, output, archive, qa)
    assert first['optional_fit051_curve_screenshots'] == ['fit051-v1-curves.png']
    previous = (output / 'inventory.json').read_bytes()
    new_qa = copy.deepcopy(qa)
    new_qa['archive_manifest_sha256'] = 'b' * 64
    result = refresh_browser(output, archive, new_qa)
    assert result['archive_qa_history'] == [qa, new_qa]
    assert result['archive_qa'] == new_qa
    assert (output / result['previous_inventory']['path']).read_bytes() == previous
    assert result['copied_files'] == first['copied_files']
    for name, expected in {**result['copied_files'], **result['generated_files']}.items():
        assert sha(output / name) == expected
    for name in result['copied_files']:
        assert (output / name).read_bytes() == (source / name).read_bytes()


def test_browser_refresh_rejects_corruption_before_rewriting_inventory(tmp_path):
    source, archive, qa = receipts(tmp_path)
    output = tmp_path / 'browser'
    preserve_browser(source, output, archive, qa)
    previous = (output / 'inventory.json').read_bytes()
    (output / 'fit051-v1-native.png').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Existing browser evidence hash'):
        refresh_browser(output, archive, qa)
    assert (output / 'inventory.json').read_bytes() == previous


def test_browser_packaging_cannot_write_into_immutable_archive(tmp_path):
    source, archive, qa = receipts(tmp_path)
    with pytest.raises(ValueError, match='Unsafe browser output'):
        preserve_browser(source, archive / 'browser', archive, qa)
    assert list(archive.iterdir()) == []
