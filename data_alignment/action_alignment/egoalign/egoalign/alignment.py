"""Validated camera-alignment metadata shared by staging and candidates."""
import math


def parse_alignment(payload, *, require_duration=False):
    try:
        offset = float(payload['offset_s'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Invalid alignment offset_s') from exc
    if not math.isfinite(offset):
        raise ValueError('Alignment offset_s must be finite')
    durations = []
    for key in ('camera_duration_s', 'gopro_duration_s'):
        if key in payload:
            try:
                value = float(payload[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Invalid alignment {key}') from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'Alignment {key} must be finite and positive')
            durations.append(value)
    if len(durations) == 2 and not math.isclose(durations[0], durations[1], rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError('Conflicting camera_duration_s and gopro_duration_s')
    if not durations and require_duration:
        raise ValueError('Alignment needs camera_duration_s or gopro_duration_s')
    duration = durations[0] if durations else None
    if duration is not None and not math.isfinite(offset + duration):
        raise ValueError('Alignment interval is not finite')
    return offset, duration
