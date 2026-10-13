# Third-party components and terms

Third-party code and assets retain their original notices and terms.

## Runtime and inherited code

- [The public runtime manifest](../runtime/patches/manifest.json) identifies the
  pinned NVlabs/GR00T-WholeBodyControl source, compatibility patches and file hashes.
  See [runtime setup](public_runtime.md) for installation and source attribution.
  Consult the notices in that upstream checkout; models are obtained separately.
- `egoalign/geometry/kornia_transform.py` is inherited from the pinned runtime's
  `gear_sonic/trl/utils/kornia_transform.py`; its contents are unchanged.
  Its source references include Kornia, QuaterNet, Ceres, transforms3d,
  TensorFlow Graphics and SfmLearner-Pytorch. Preserve the references and notices.
  Kornia publishes a [copyright inventory](https://github.com/kornia/kornia/blob/main/COPYRIGHT)
  and [license](https://github.com/kornia/kornia/blob/main/LICENSE).

Preserve source-level copyright, license and attribution notices when modifying
or redistributing these components.

## External dependencies and assets

[Qualified dependencies](DEPENDENCIES.md) records the tested versions and source
hashes; external dependency source distributions retain their upstream terms.
BrainCo meshes, SMPL parameters and controller weights are not bundled. Obtain
them from their rights holders and configure [external asset paths](EXTERNAL_ASSETS.md).

