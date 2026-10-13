# Object Relocation sample data

`data/basket/` contains one Object Relocation episode and its two camera videos:
compact episode zero, numerical body/hand observations, minimal LeRobot metadata,
existing camera-alignment results and a checksummed `sample_manifest.json`.
The recorded instruction is `Move the basket to the table`.

## Content and integrity

The sample has 1009 raw frames; the inherited camera-overlap interval selects
1006 frames. Numerical Parquet values are preserved, while source schema metadata
is removed. MP4s are remuxed without re-encoding: video packet hashes, PTS, DTS
and durations are verified against the source. Audio/data tracks, chapters and
source container metadata are omitted; visible scene content is unchanged.
Container hashes consequently differ from the original videos.

No other episodes, capture logs, camera serials, network credentials, user paths
or GPMF sidecars are included. `python tools/audit_source.py` verifies each included
manifest member and rejects unapproved sample directories. The manifest records
the source-equivalence checks made during sample packaging.

These are raw reproduction inputs. The [recipe](README.md) reconstructs causal
state and motion-token supervision through actual controller replay.

