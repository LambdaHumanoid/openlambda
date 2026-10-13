import hashlib
import pytest
from egoalign.runtime_assets import materialize


def test_assets_verify_and_refuse_overwrite(tmp_path):
    target = tmp_path / 'model'
    target.write_bytes(b'model')
    manifest = {'assets': [{'destination': 'model', 'sha256': hashlib.sha256(b'model').hexdigest()}]}
    assert len(materialize(tmp_path, manifest)) == 1
    target.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='No overwrite'):
        materialize(tmp_path, manifest, download=True)
    assert target.read_bytes() == b'wrong'


def test_assets_missing_and_escape(tmp_path):
    with pytest.raises(ValueError, match='Missing'):
        materialize(tmp_path, {'assets': [{'destination': 'missing', 'sha256': 'x'}]})
    with pytest.raises(ValueError, match='escapes'):
        materialize(tmp_path, {'assets': [{'destination': '../escape', 'sha256': 'x'}]})
