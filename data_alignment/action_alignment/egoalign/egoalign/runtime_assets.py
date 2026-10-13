"""Verify pinned public SONIC assets; download missing files only on explicit request."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

MANIFEST = Path(__file__).resolve().parents[1] / "runtime/assets.json"


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def materialize(root, manifest, download=False):
    records = []
    for asset in manifest['assets']:
        target = (root / asset['destination']).resolve()
        if root not in target.parents:
            raise ValueError('Asset destination escapes runtime checkout')
        if target.exists():
            if digest(target) != asset['sha256']:
                raise ValueError(f'Existing asset differs (or is an LFS pointer): {target}; '
                                 'back it up explicitly before retrying. No overwrite performed.')
        elif not download:
            raise ValueError(f'Missing asset: {target}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            url = (f"https://huggingface.co/{manifest['repository']}/resolve/"
                   f"{manifest['revision']}/{asset['name']}")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.download-', delete=False) as out:
                    temporary = Path(out.name)
                    with urlopen(url, timeout=120) as response:
                        for chunk in iter(lambda: response.read(1024 * 1024), b''):
                            out.write(chunk)
                if digest(temporary) != asset['sha256']:
                    raise ValueError(f'Download checksum mismatch: {asset["name"]}')
                # Hard-link publish refuses concurrent replacement and is same-filesystem.
                target.hardlink_to(temporary)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        records.append({'path': str(target), 'sha256': asset['sha256']})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='Accept upstream license and download missing files')
    args = parser.parse_args(argv)
    try:
        root = args.runtime_root.resolve()
        if not root.is_dir():
            raise ValueError('Runtime checkout must exist')
        print(json.dumps(materialize(root, json.loads(MANIFEST.read_text()), args.download), indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Runtime assets: {exc}\n')


if __name__ == '__main__':
    main()
