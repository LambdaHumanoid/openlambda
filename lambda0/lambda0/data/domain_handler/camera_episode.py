"""Sample once per camera frame while predicting an independent 50 Hz action stream."""
import random
import av
import numpy as np
import torch
from PIL import Image
from .parquet_episode import ParquetEpisodeHandler
from ..camera_timeline import load_camera_episode, future_action_indices, episode_path, camera_valid_observations
from ..video import decode


class CameraEpisodeHandler(ParquetEpisodeHandler):
    def _camera_episode(self, episode_index):
        item = self.meta["datalist"][episode_index]
        return item, load_camera_episode(self.meta, item)

    def _camera_sample(self, data, index, frame, instruction, num_actions, image_aug):
        state, action, sm, am, ct, at, _ = data
        indices = future_action_indices(ct[index:index+1], at, num_actions)[0]
        if not sm[index].any() or not am[indices].any():
            return None
        proprio, target = state[index].copy(), action[indices].copy()
        pm, tm = sm[index].copy(), am[indices].copy()
        if self.normalizer is not None:
            proprio = self.normalizer.normalize_proprio(proprio, pm)
            target = self.normalizer.normalize_action(target, tm)
        image = image_aug(Image.fromarray(frame))
        images = [image] + [torch.zeros_like(image) for _ in range(self.num_views - 1)]
        image_mask = torch.zeros(self.num_views, dtype=torch.bool); image_mask[0] = True
        return dict(language_instruction=instruction, image_input=torch.stack(images),
                    image_mask=image_mask, proprio=torch.from_numpy(proprio),
                    proprio_valid_mask=torch.from_numpy(pm), action=torch.from_numpy(target),
                    action_valid_mask=torch.from_numpy(tm), sample_index=index)

    def sample_episode(self, episode_index, *, num_actions, image_aug, lang_aug_map, sample_index=None):
        item, data = self._camera_episode(episode_index)
        state, action, sm, am, ct, at, video_indices = data
        candidates = camera_valid_observations(sm, am, ct, at, num_actions)
        if sample_index is None:
            if not candidates: return None
            index = random.choice(candidates)
        else:
            index = int(sample_index)
            if not 0 <= index < len(state): raise IndexError(index)
            if index not in candidates: return None
        frame = decode(episode_path(self.meta, item, "video_path"), [int(video_indices[index])])[0]
        return self._camera_sample(data, index, frame, self._instruction(item, lang_aug_map), num_actions, image_aug)

    def iter_episode(self, episode_index, *, num_actions, training, image_aug, lang_aug_map, **kwargs):
        item, data = self._camera_episode(episode_index)
        _, _, sm, am, ct, at, video_indices = data
        instruction = self._instruction(item, lang_aug_map)
        candidates = camera_valid_observations(sm, am, ct, at, num_actions)
        wanted = {int(video_indices[i]): i for i in candidates}
        if not wanted:
            return
        remaining = len(wanted)
        with av.open(str(episode_path(self.meta, item, "video_path"))) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                index = wanted.get(frame_index)
                if index is not None:
                    yield self._camera_sample(data, index, frame.to_ndarray(format="rgb24"), instruction, num_actions, image_aug)
                    remaining -= 1
                    if not remaining:
                        break
        if remaining:
            raise ValueError("Video ended before all selected camera observations.")

    def iter_validation_episode(self, episode_index, *, num_actions, samples_per_episode, image_aug, lang_aug_map):
        if samples_per_episode <= 0: raise ValueError("samples_per_episode must be positive")
        item, data = self._camera_episode(episode_index)
        _, _, sm, am, ct, at, video_indices = data
        candidates = camera_valid_observations(sm, am, ct, at, num_actions)
        if not candidates: return
        count = min(samples_per_episode, len(candidates))
        offsets = ((np.arange(count) + .5) * len(candidates) / count).astype(int)
        for i in offsets:
            index = candidates[i]
            frame = decode(episode_path(self.meta, item, "video_path"), [int(video_indices[index])])[0]
            yield self._camera_sample(data, index, frame, self._instruction(item, lang_aug_map), num_actions, image_aug)
