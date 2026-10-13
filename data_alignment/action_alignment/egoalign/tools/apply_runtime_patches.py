"""Check/apply the source-only runtime bundle to its exact public upstream pin."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def validate_files(root, records, key):
    errors = []
    for record in records:
        name = Path(record["path"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Unsafe manifest path")
        if digest(root / name) != record[key]:
            errors.append(str(name))
    if errors:
        raise ValueError(f"{key} mismatch: " + ", ".join(errors))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--apply", action="store_true", help="Apply after all checks; default only checks")
    p.add_argument("--verify-applied", action="store_true", help="Verify patched source hashes, without mutation")
    a = p.parse_args(argv)
    if a.apply and a.verify_applied:
        p.error("Choose apply or verify-applied")
    bundle = Path(__file__).resolve().parents[1] / "runtime/patches"
    manifest = json.loads((bundle / "manifest.json").read_text())
    root = a.runtime_root.resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.STDOUT).decode().strip()
    try:
        if git("rev-parse", "HEAD") != manifest["upstream_commit"]:
            raise ValueError("Wrong upstream commit; use the documented detached checkout")
        patches = []
        for record in manifest["patches"]:
            path = bundle / record["path"]
            if path.parent != bundle or digest(path) != record["sha256"]:
                raise ValueError("Patch checksum/path mismatch")
            patches.append(str(path))
        if a.verify_applied:
            validate_files(root, manifest["files"], "after_sha256")
            print("PASS: applied source hashes (not a build/replay test)")
            return 0
        if git("status", "--porcelain", "--untracked-files=normal"):
            raise ValueError("Refusing a dirty source checkout; use a separate fresh checkout")
        validate_files(root, manifest["files"], "before_sha256")
        git("apply", "--check", *patches)
        if a.apply:
            git("apply", *patches)
            validate_files(root, manifest["files"], "after_sha256")
        print("PASS: " + ("applied and hash-verified" if a.apply else "application check only"))
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        detail = exc.output.decode(errors="replace") if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        p.exit(2, f"Runtime patch: {detail}\n")


if __name__ == "__main__":
    raise SystemExit(main())
