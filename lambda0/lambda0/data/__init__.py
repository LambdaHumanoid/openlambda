# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader


def worker_init_fn(worker_id: int) -> None:
    base_seed = torch.initial_seed() % (2**32)
    import random

    import numpy as np

    np.random.seed(base_seed)
    random.seed(base_seed)
    torch.manual_seed(base_seed)


def create_dataloader(
    batch_size: int,
    metas_path: str,
    num_actions: int,
    training: bool,
    num_workers: int = 4,
    dataset_names: list[str] | None = None,
    sample_weights: dict[str, float] | None = None,
    sampling_plan: Mapping[str, Any] | None = None,
    sampling_seed: int = 0,
    validation_samples_per_episode: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    data_root: str | None = None,
    image_max_pixels: int | None = None,
):
    from .dataset import InfiniteDataReader
    from .heterogeneous_collator import HeterogeneousActionCollator

    return DataLoader(
        InfiniteDataReader(
            metas_path,
            num_actions=num_actions,
            training=training,
            dataset_names=dataset_names,
            sample_weights=sample_weights,
            sampling_plan=sampling_plan,
            sampling_seed=sampling_seed,
            validation_samples_per_episode=validation_samples_per_episode,
            shard_index=shard_index,
            num_shards=num_shards,
            data_root=data_root,
            image_max_pixels=image_max_pixels,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=HeterogeneousActionCollator(),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )
