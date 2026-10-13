"""Read-only check of user-supplied BrainCo assets; no download or substitution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def inventory(root):
    root = root.resolve()
    files, visited = set(), set()

    def required(path):
        path = path.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing/outside asset root: {path.name}')
        if path.read_bytes()[:42].startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'Unmaterialized Git LFS pointer: {path.name}')
        files.add(path)
        return path

    def walk(path):
        path = required(path)
        if path in visited:
            return
        visited.add(path)
        tree = ET.parse(path).getroot()
        compiler = tree.find('compiler')
        meshdir = compiler.get('meshdir', '') if compiler is not None else ''
        texturedir = compiler.get('texturedir', '') if compiler is not None else ''
        for node in tree.iter():
            if node.tag == 'include':
                walk(path.parent / node.attrib['file'])
            elif node.tag in ('mesh', 'texture', 'hfield') and node.get('file'):
                subdir = meshdir if node.tag == 'mesh' else texturedir if node.tag == 'texture' else ''
                required(path.parent / subdir / node.attrib['file'])

    for name in ('scene_29dof_brainco.xml', 'g1_29dof_brainco.xml'):
        walk(root / name)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=os.environ.get('EGOALIGN_BRAINCO_MODEL_DIR'))
    p.add_argument('--expected', type=Path, help='Optional exact file-hash manifest from a qualified run')
    p.add_argument('--load-mujoco', action='store_true', help='Also compile MJCF, without stepping simulation')
    a = p.parse_args()
    if a.model_dir is None:
        p.error('Set EGOALIGN_BRAINCO_MODEL_DIR or pass --model-dir; assets are not bundled')
    try:
        files = inventory(a.model_dir)
        if a.expected and files != json.loads(a.expected.read_text())['files']:
            raise ValueError('Asset hashes differ from the supplied reference; qualify this model separately')
        report = {'files': files, 'passed': True, 'mujoco_loaded': False,
                  'scope': 'Asset presence/hash check, not license or dynamics certification'}
        if a.load_mujoco:
            import mujoco
            m = mujoco.MjModel.from_xml_path(str(a.model_dir / 'scene_29dof_brainco.xml'))
            report.update(mujoco_loaded=True, nq=m.nq, nv=m.nv, nu=m.nu)
        print(json.dumps(report, indent=2))
    except (ValueError, OSError, ET.ParseError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
