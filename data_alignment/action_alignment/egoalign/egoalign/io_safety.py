"""Fail closed before creating or replacing dataset output directories."""
from pathlib import Path
from typing import Iterable


def validate_output_root(output: Path, inputs: Iterable[Path] = ()) -> Path:
    """Return a canonical output path, rejecting broad or overlapping targets.

    Both directions of nesting are rejected. Resolve symlinks before comparing;
    do not use this helper as a substitute for permission/concurrency isolation.
    """
    output = Path(output).expanduser().resolve()
    protected = (Path.home().resolve(), Path.cwd().resolve(), Path(__file__).resolve().parents[1])
    if any(output == root or output in root.parents for root in protected):
        raise ValueError(f"Refusing broad output directory: {output}")
    for source in inputs:
        source = Path(source).expanduser().resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"Input and output paths must be disjoint: {source} / {output}")
    return output
