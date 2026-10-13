import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from egoalign.check_robot_assets import inventory


def test_assets_missing_and_external_rejected(tmp_path):
    with pytest.raises(ValueError, match='Missing'):
        inventory(tmp_path)
    (tmp_path / 'scene_29dof_brainco.xml').write_text('<mujoco><include file="../outside.xml"/></mujoco>')
    with pytest.raises(ValueError, match='outside'):
        inventory(tmp_path)


def test_assets_inventory_and_mesh(tmp_path):
    (tmp_path / 'scene_29dof_brainco.xml').write_text('<mujoco><include file="g1_29dof_brainco.xml"/></mujoco>')
    (tmp_path / 'g1_29dof_brainco.xml').write_text('<mujoco><compiler meshdir="meshes"/><asset><mesh file="hand.stl"/></asset></mujoco>')
    with pytest.raises(ValueError, match='hand.stl'):
        inventory(tmp_path)
    (tmp_path / 'meshes').mkdir()
    (tmp_path / 'meshes/hand.stl').write_bytes(b'mesh')
    assert len(inventory(tmp_path)) == 3


def test_sample_audit_rejects_tampered_and_unlisted_binary(tmp_path):
    script = tmp_path / 'tools/audit_source.py'
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).parents[1] / 'tools/audit_source.py', script)
    sample = tmp_path / 'examples/data/basket'
    sample.mkdir(parents=True)
    payload = sample / 'camera.mp4'
    payload.write_bytes(b'test')
    (sample / 'sample_manifest.json').write_text(json.dumps({'files': [{
        'path': 'camera.mp4', 'bytes': 4, 'sha256': hashlib.sha256(b'test').hexdigest()}]}))
    def run():
        return subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert run().returncode == 0
    payload.write_bytes(b'edit')
    assert 'Sample hash/size mismatch' in run().stdout
    payload.write_bytes(b'test')
    (sample / 'unexpected.mp4').write_bytes(b'test')
    assert 'Unexpected payload type' in run().stdout


def test_sample_audit_rejects_escaping_manifest(tmp_path):
    script = tmp_path / 'tools/audit_source.py'
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).parents[1] / 'tools/audit_source.py', script)
    sample = tmp_path / 'examples/data/basket'
    sample.mkdir(parents=True)
    (sample / 'sample_manifest.json').write_text(json.dumps({'files': [{
        'path': '../outside.mp4', 'bytes': 0, 'sha256': ''}]}))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Invalid sample member' in result.stdout


def test_sample_audit_rejects_unapproved_sample_even_with_valid_manifest(tmp_path):
    script = tmp_path / 'tools/audit_source.py'
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).parents[1] / 'tools/audit_source.py', script)
    sample = tmp_path / 'examples/data/pedal'
    sample.mkdir(parents=True)
    (sample / 'sample_manifest.json').write_text(json.dumps({'files': []}))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Unapproved sample payload: examples/data/pedal' in result.stdout


@pytest.mark.parametrize('directory', ['internal', 'validation'])
def test_source_audit_rejects_private_workspace_directory(tmp_path, directory):
    script = tmp_path / 'tools/audit_source.py'
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).parents[1] / 'tools/audit_source.py', script)
    (tmp_path / directory).mkdir()
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode != 0
    assert f'Private workspace directory in distribution: {directory}' in result.stdout


def run_source_audit(tmp_path):
    script = tmp_path / 'tools/audit_source.py'
    script.parent.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__).parents[1] / 'tools/audit_source.py', script)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    return result.returncode, json.loads(result.stdout)


@pytest.mark.parametrize('relative', ['credentials.json', '.release-secrets.json',
                                    '.private/credentials.json', '.gitignore'])
def test_source_audit_checks_visible_and_hidden_sensitive_content(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('-----BEGIN ' + 'OPENSSH PRIVATE KEY-----\nNOT_A_REAL_KEY\n')
    code, report = run_source_audit(tmp_path)
    assert code != 0
    assert f'Local identity/address or private key marker: {relative}' in report['failures']


@pytest.mark.parametrize('relative', ['.env', '.env.local', '.ssh/id_rsa', '.hidden/payload.bin'])
def test_source_audit_rejects_unexpected_hidden_payloads(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'synthetic payload')
    code, report = run_source_audit(tmp_path)
    assert code != 0
    assert f'Unexpected payload type: {relative}' in report['failures']


@pytest.mark.parametrize('relative', ['.linked.json', '.hidden/linked.json', '.venv', '.pytest_cache'])
@pytest.mark.parametrize('kind', ['file', 'directory', 'dangling'])
def test_source_audit_rejects_hidden_symlinks(tmp_path, relative, kind):
    target = tmp_path / 'target.json'
    if kind == 'directory':
        target.mkdir()
    elif kind == 'file':
        target.write_text('{}')
    link = tmp_path / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=kind == 'directory')
    code, report = run_source_audit(tmp_path)
    assert code != 0
    assert f'Symlink is not distributable: {relative}' in report['failures']


def test_source_audit_inventories_harmless_dotfiles(tmp_path):
    for name, content in (('.settings.json', '{}'), ('.gitignore', 'outputs/\n')):
        (tmp_path / name).write_text(content)
    code, report = run_source_audit(tmp_path)
    assert code == 0
    files = {entry['path']: entry['sha256'] for entry in report['files']}
    assert files['.settings.json'] == hashlib.sha256(b'{}').hexdigest()
    assert '.gitignore' in files


@pytest.mark.parametrize('relative', ['.git', '.venv', '.pytest_cache', '.mypy_cache',
                                    '.ruff_cache', '__pycache__', 'nested/__pycache__'])
def test_source_audit_records_explicit_local_exclusions(tmp_path, relative):
    directory = tmp_path / relative
    directory.mkdir(parents=True)
    (directory / 'cache.bin').write_bytes(b'local only')
    code, report = run_source_audit(tmp_path)
    assert code == 0
    assert report['excluded_local_paths'] == [relative]


def test_source_audit_records_git_worktree_file_exclusion(tmp_path):
    (tmp_path / '.git').write_text('gitdir: ../metadata\n')
    code, report = run_source_audit(tmp_path)
    assert code == 0
    assert report['excluded_local_paths'] == ['.git']


def test_source_audit_inventories_upstream_root_license(tmp_path):
    (tmp_path / 'LICENSE').write_text('Apache License\nVersion 2.0\n')
    code, report = run_source_audit(tmp_path)
    assert code == 0
    assert 'LICENSE' in {entry['path'] for entry in report['files']}


def test_source_audit_still_scans_license_content(tmp_path):
    (tmp_path / 'LICENSE').write_text('-----BEGIN ' + 'OPENSSH PRIVATE KEY-----\n')
    code, report = run_source_audit(tmp_path)
    assert code != 0
    assert 'Local identity/address or private key marker: LICENSE' in report['failures']


@pytest.mark.parametrize('filename', ['figures/pipeline.png', 'figures/title_light.svg',
                                    'figures/title_dark.svg', 'videos/storage_box_comparison.mp4',
                                    'videos/shelf_interaction_comparison.mp4',
                                    'videos/parcel_delivery_comparison.mp4',
                                    'videos/indoor_cleaning_comparison.mp4'])
@pytest.mark.parametrize('damage', ['none', 'tampered', 'unlisted', 'traversal'])
def test_documentation_media_manifest(tmp_path, damage, filename):
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / filename).parent.mkdir(parents=True, exist_ok=True)
    (assets / 'metrics').mkdir()
    (assets / filename).write_bytes(b'preview')
    record = {'path': filename, 'bytes': 7,
              'sha256': hashlib.sha256(b'preview').hexdigest()}
    if damage == 'tampered':
        (assets / filename).write_bytes(b'changed')
    elif damage == 'unlisted':
        (assets / 'private.mp4').write_bytes(b'private')
    elif damage == 'traversal':
        record['path'] = '../' + filename
    (assets / 'metrics/media_manifest.json').write_text(json.dumps({'files': [record]}))
    code, report = run_source_audit(tmp_path)
    assert (code == 0) == (damage == 'none'), report
