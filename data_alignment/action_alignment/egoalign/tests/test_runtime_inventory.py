import hashlib

from egoalign.runtime_inventory import file_record, missing_libraries, python_record


def test_missing_libraries():
    assert missing_libraries("liba.so => not found\nlibb.so => /lib/b\nliba.so => not found") == ["liba.so"]


def test_hash_and_missing(tmp_path):
    path = tmp_path / "asset"
    assert file_record(path) == {"present": False}
    path.write_bytes(b"asset")
    assert file_record(path) == {"present": True, "bytes": 5,
                                 "sha256": hashlib.sha256(b"asset").hexdigest()}


def test_failed_interpreter(tmp_path):
    assert python_record(tmp_path / "absent")["returncode"] == -1
