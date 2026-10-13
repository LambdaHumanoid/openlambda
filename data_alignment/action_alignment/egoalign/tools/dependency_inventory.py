"""Print dependency provenance and installed LeRobot Python source hashes."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path


def inventory():
    names = ('torch', 'torchvision', 'numpy', 'scipy', 'pin', 'pandas', 'pyarrow',
             'datasets', 'lerobot', 'mujoco', 'onnxruntime', 'av', 'pyzmq')
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    dist = metadata.distribution('lerobot')
    direct = json.loads(dist.read_text('direct_url.json') or '{}')
    # Only public VCS provenance is included, never editable local install paths.
    provenance = {'url': direct.get('url'), 'vcs_info': direct.get('vcs_info')}
    if not str(provenance['url']).startswith('https://github.com/huggingface/lerobot'):
        provenance = {'note': 'Non-public install location omitted'}
    records = []
    for rel in sorted(dist.files or [], key=str):
        if str(rel).startswith('lerobot/') and str(rel).endswith('.py'):
            path = Path(dist.locate_file(rel))
            records.append({'path': str(rel), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'scope': 'observed installation; hashes identify local content, not proof of upstream equality',
            'versions': versions, 'lerobot_provenance': provenance, 'lerobot_python_files': records}


if __name__ == '__main__':
    print(json.dumps(inventory(), indent=2))
