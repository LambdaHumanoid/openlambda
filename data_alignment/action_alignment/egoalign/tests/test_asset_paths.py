from egoalign.runtime import human_skeleton_path
import pytest

def test_asset_root_independent_of_cwd(tmp_path, monkeypatch):
    root = tmp_path / 'assets'
    expected = root / 'gear_sonic/data/human/human_joints_info.pkl'
    expected.parent.mkdir(parents=True)
    expected.touch()
    monkeypatch.delenv('EGOALIGN_HUMAN_JOINTS_PATH', raising=False)
    monkeypatch.setenv('EGOALIGN_ASSET_ROOT', str(root))
    monkeypatch.chdir(tmp_path)
    assert human_skeleton_path() == expected

def test_explicit_file_override(tmp_path, monkeypatch):
    expected = tmp_path / 'skeleton.pkl'
    expected.touch()
    monkeypatch.setenv('EGOALIGN_HUMAN_JOINTS_PATH', str(expected))
    monkeypatch.setenv('EGOALIGN_ASSET_ROOT', str(tmp_path / 'missing'))
    assert human_skeleton_path() == expected

def test_missing_asset_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv('EGOALIGN_HUMAN_JOINTS_PATH', str(tmp_path / 'missing.pkl'))
    with pytest.raises(FileNotFoundError, match='EGOALIGN_HUMAN_JOINTS_PATH'):
        human_skeleton_path()
