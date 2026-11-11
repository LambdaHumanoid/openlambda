"""Expose checkout policies alongside an installed SIMPLE package."""

from importlib.util import spec_from_file_location as _spec_from_file_location
from pathlib import Path as _Path
from pkgutil import extend_path as _extend_path

__path__ = _extend_path(__path__, __name__)
_overlay_init = _Path(__file__).resolve()
for _package_path in __path__:
    _native_init = _Path(_package_path) / "__init__.py"
    if _native_init.is_file() and _native_init.resolve() != _overlay_init:
        # Resource lookup must use the native loader, not just native __file__.
        # Keep the combined search path so checkout baselines remain importable.
        __file__ = str(_native_init)
        __spec__ = _spec_from_file_location(
            __name__, __file__, submodule_search_locations=__path__
        )
        __loader__ = __spec__.loader
        __cached__ = __spec__.cached
        exec(compile(_native_init.read_bytes(), __file__, "exec"), globals())
        break
else:
    raise ImportError(
        "The OpenLambda SIMPLE adapters require an installed SIMPLE package. "
        "Keep SIMPLE's src directory or site-packages on PYTHONPATH."
    )

del _Path, _extend_path, _spec_from_file_location, _overlay_init, _package_path, _native_init
