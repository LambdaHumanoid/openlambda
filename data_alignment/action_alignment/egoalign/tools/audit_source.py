"""Read-only source audit and provenance report; not a full replay validator."""
import ast
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-manifest', type=Path, help='Optional private maintainer provenance file')
args = parser.parse_args()
source_records = ({r['path']: r for r in json.loads(args.source_manifest.read_text())}
                  if args.source_manifest else {})
failures, files, excluded = [], [], []
# These local development directories are outside the distributable source tree.
# Do not skip arbitrary dotfiles: they can contain credentials or private paths.
CACHE_DIRECTORIES = {'__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'}
ROOT_LOCAL_DIRECTORIES = {'.git', '.venv'}


def source_paths():
    for directory, directories, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in sorted(directories):
            path = parent / name
            relative = path.relative_to(root)
            if path.is_symlink():
                failures.append(f'Symlink is not distributable: {relative}')
                directories.remove(name)
            elif name in CACHE_DIRECTORIES or (parent == root and name in ROOT_LOCAL_DIRECTORIES):
                excluded.append(str(relative))
                directories.remove(name)
        for name in sorted(filenames):
            path = parent / name
            relative = path.relative_to(root)
            if path.is_symlink():
                failures.append(f'Symlink is not distributable: {relative}')
            elif parent == root and name == '.git':  # Git worktree metadata file.
                excluded.append(str(relative))
            elif path.is_file():
                yield path


# Release preparation records must never be copied into the distributable tree.
for private_directory in ('internal', 'validation'):
    if (root / private_directory).exists():
        failures.append(f'Private workspace directory in distribution: {private_directory}')
sample_files = {}
# Publication approval is basket-only. A manifest is not publication permission.
approved_sample_names = {'basket'}
sample_root = root / 'examples/data'
if sample_root.exists():
    for member in sorted(sample_root.iterdir()):
        if member.name not in approved_sample_names:
            failures.append(f'Unapproved sample payload: {member.relative_to(root)}')
for manifest in sorted((root / 'examples/data').glob('*/sample_manifest.json')):
    base = manifest.parent
    if base.name not in approved_sample_names:
        continue
    for record in json.loads(manifest.read_text())['files']:
        rel = Path(record['path'])
        path = base / rel
        if (rel.is_absolute() or '..' in rel.parts or path.is_symlink() or
                any(parent.is_symlink() for parent in path.parents if parent != root) or
                not path.is_file()):
            failures.append(f'Invalid sample member: {rel}')
            continue
        if (path.stat().st_size != record['bytes'] or
                hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']):
            failures.append(f'Sample hash/size mismatch: {path.relative_to(root)}')
        sample_files[path] = record
media_files = {}
approved_media = {'figures/pipeline.png', 'figures/title_light.svg', 'figures/title_dark.svg',
                  'videos/object_relocation_comparison.mp4',
                  'videos/storage_box_comparison.mp4', 'videos/shelf_interaction_comparison.mp4',
                  'videos/parcel_delivery_comparison.mp4', 'videos/indoor_cleaning_comparison.mp4'}
media_manifest = root / 'assets/metrics/media_manifest.json'
if media_manifest.exists():
    for record in json.loads(media_manifest.read_text())['files']:
        rel = Path(record['path'])
        path = root / 'assets' / rel
        if (str(rel) not in approved_media or path.is_symlink() or
                not path.is_file() or any(p.is_symlink() for p in path.parents if p != root)):
            failures.append(f'Unapproved documentation media: {rel}')
            continue
        if (path.stat().st_size != record['bytes'] or
                hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']):
            failures.append(f'Documentation media hash/size mismatch: {rel}')
        media_files[path] = record
for path in sorted(source_paths()):
    if path in media_files:
        files.append({'path': str(path.relative_to(root)), 'sha256': media_files[path]['sha256']})
        continue
    if path.suffix in {'.parquet', '.mp4'} and path in sample_files:
        files.append({'path': str(path.relative_to(root)), 'sha256': sample_files[path]['sha256']})
        continue
    if path.suffix not in {'.py', '.md', '.toml', '.sh', '.json', '.patch'} and path.name not in {'UPSTREAM_LICENSE', '.gitignore'} and path != root / 'LICENSE' and not (path.suffix == '.jsonl' and path in sample_files):
        failures.append(f'Unexpected payload type: {path.relative_to(root)}')
        continue
    if path.name in {'source_audit.json'}:
        continue
    content = path.read_text(errors='replace')
    if path.suffix == '.py':
        ast.parse(content, filename=str(path))
    # Removed/context lines in a patch come from the pinned public upstream.
    # Audit additions too; do not waive the runtime bundle as a whole.
    audited = ('\n'.join(line for line in content.splitlines() if line.startswith('+'))
               if path.suffix == '.patch' else content)
    if re.search(r'/home/[A-Za-z0-9_-]+/|192\.168\.\d+\.\d+|8\.141\.\d+\.\d+|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY', audited):
        failures.append(f'Local identity/address or private key marker: {path.relative_to(root)}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    original = source_records.get(str(path.relative_to(root)), {}).get('source_sha256')
    files.append({'path': str(path.relative_to(root)), 'sha256': digest,
                  'modified_from_extracted_source': bool(original and original != digest)})
report = {'scope': 'static source packaging only; no runtime/replay qualification',
          'passed': not failures, 'failures': failures, 'files': files,
          'excluded_local_paths': sorted(excluded)}
print(json.dumps(report, indent=2))
raise SystemExit(bool(failures))
