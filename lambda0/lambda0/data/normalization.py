"""Serialized, dataset-specific normalization for heterogeneous actions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class DatasetNormalizer:
    """Apply a validated affine normalization to actions and proprioception."""

    def __init__(
        self,
        path: str | Path,
        dataset_name: str | None = None,
        *,
        expected_action_space: str | None = None,
        expected_coordinate_contract: str | None = None,
        expected_normalization_contract: str | None = None,
        action_dim: int | None = None,
        proprio_dim: int | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Normalization file does not exist: {self.path}")
        with self.path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise TypeError(f"Normalization file must contain an object: {self.path}")
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError(f"Unsupported normalization schema: {self.path}")
        if payload.get("release_contract") != "lambda0_velocity_v1" or payload.get("method") != "q01_q99":
            raise ValueError("Release requires train-split q01/q99 statistics from prepare_release_data.py.")

        self.dataset_name = str(payload.get("dataset_name", ""))
        self.action_space = str(payload.get("action_space", ""))
        self.coordinate_contract = str(payload.get("coordinate_contract", ""))
        self.normalization_contract = str(
            payload.get("normalization_contract", "")
        )
        if not self.dataset_name or not self.action_space:
            raise ValueError(
                "Normalization dataset_name and action_space must be non-empty."
            )
        self._validate_identity(
            dataset_name=dataset_name,
            action_space=expected_action_space,
            coordinate_contract=expected_coordinate_contract,
            normalization_contract=expected_normalization_contract,
        )

        self.action_location, self.action_scale = self._parameters(
            payload,
            "action",
            action_dim,
        )
        self.proprio_location, self.proprio_scale = self._parameters(
            payload,
            "proprio",
            proprio_dim,
        )
        self.action_dim = int(self.action_location.size)
        self.proprio_dim = int(self.proprio_location.size)
        self.action_q01 = np.asarray(payload["action"]["q01"], dtype=np.float32)
        self.proprio_q01 = np.asarray(payload["proprio"]["q01"], dtype=np.float32)
        self.clip = None  # The release contract never clips normalized values.

    def _validate_identity(
        self,
        *,
        dataset_name: str | None,
        action_space: str | None,
        coordinate_contract: str | None,
        normalization_contract: str | None,
    ) -> None:
        expected = {
            "dataset_name": dataset_name,
            "action_space": action_space,
            "coordinate_contract": coordinate_contract,
            "normalization_contract": normalization_contract,
        }
        actual = {
            "dataset_name": self.dataset_name,
            "action_space": self.action_space,
            "coordinate_contract": self.coordinate_contract,
            "normalization_contract": self.normalization_contract,
        }
        for key, expected_value in expected.items():
            if expected_value is not None and actual[key] != expected_value:
                raise ValueError(
                    f"Normalizer {key} {actual[key]!r} does not match "
                    f"{expected_value!r}."
                )

    @staticmethod
    def _parameters(
        payload: dict,
        key: str,
        expected_dim: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        block = payload.get(key)
        if not isinstance(block, dict):
            raise KeyError(f"Normalization payload has no {key!r} block.")
        location = np.asarray(block.get("location"), dtype=np.float32)
        scale = np.asarray(block.get("scale"), dtype=np.float32)
        if location.ndim != 1 or scale.shape != location.shape or not location.size:
            raise ValueError(
                f"{key} normalization location and scale must be equal 1D arrays."
            )
        if expected_dim is not None and location.shape != (int(expected_dim),):
            raise ValueError(
                f"{key} normalization has {location.size} dimensions, "
                f"expected {expected_dim}."
            )
        declared_dim = payload.get(f"{key}_dim")
        if declared_dim is not None and int(declared_dim) != location.size:
            raise ValueError(
                f"{key}_dim={declared_dim} does not match {location.size} parameters."
            )
        if not np.isfinite(location).all() or not np.isfinite(scale).all():
            raise ValueError(f"{key} normalization contains non-finite values.")
        if bool((scale <= 0).any()):
            raise ValueError(f"{key} normalization scales must be positive.")
        return location, scale

    @staticmethod
    def _validated_value_and_mask(
        value: np.ndarray,
        valid_mask: np.ndarray | None,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim < 1 or array.shape[-1] != width:
            raise ValueError(
                f"Normalization value must end in {width} dimensions, "
                f"got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError("Normalization value contains non-finite values.")
        valid = (
            np.ones(array.shape, dtype=bool)
            if valid_mask is None
            else np.asarray(valid_mask, dtype=bool)
        )
        if valid.shape != array.shape:
            raise ValueError("Normalization value and mask shapes are incompatible.")
        return array, valid

    def _normalize(
        self,
        value: np.ndarray,
        valid_mask: np.ndarray | None,
        location: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        array, valid = self._validated_value_and_mask(
            value,
            valid_mask,
            int(location.size),
        )
        # Here location is q01, not the affine midpoint. This ordering also
        # matches Pi0.5 for large constant dimensions (no midpoint cancellation).
        normalized = (array - location) / (2 * scale) * 2 - 1
        normalized[~valid] = 0.0
        return normalized.astype(np.float32, copy=False)

    def _denormalize(
        self,
        value: np.ndarray,
        valid_mask: np.ndarray | None,
        location: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        array, valid = self._validated_value_and_mask(
            value,
            valid_mask,
            int(location.size),
        )
        denormalized = (array + 1) / 2 * (2 * scale) + location
        denormalized[~valid] = 0.0
        return denormalized.astype(np.float32, copy=False)

    def normalize_action(
        self,
        action: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._normalize(
            action,
            valid_mask,
            self.action_q01,
            self.action_scale,
        )

    def normalize_proprio(
        self,
        proprio: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._normalize(
            proprio,
            valid_mask,
            self.proprio_q01,
            self.proprio_scale,
        )

    def denormalize_action(
        self,
        action: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._denormalize(
            action,
            valid_mask,
            self.action_q01,
            self.action_scale,
        )

    def denormalize_proprio(
        self,
        proprio: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._denormalize(
            proprio,
            valid_mask,
            self.proprio_q01,
            self.proprio_scale,
        )


__all__ = ["DatasetNormalizer"]
