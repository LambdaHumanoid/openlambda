"""Pi0.5-style training augmentation before the native Qwen processor.

Preserves the source aspect ratio. Torchvision implements the same augmentation
recipe, not bit-for-bit equivalence with OpenPI's JAX/augmax kernels.
"""

import math
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class ImageAugmentation:
    def __init__(self, training: bool, max_pixels: int | None = None):
        if max_pixels is not None and (isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels <= 0):
            raise ValueError("max_pixels must be a positive integer or None")
        self.max_pixels = max_pixels
        self.training = training
        self.rotation = transforms.RandomRotation(5, interpolation=InterpolationMode.BILINEAR)
        self.color = transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)

    def __call__(self, image, *, camera_name: str = "main"):
        width, height = image.size
        if self.max_pixels is not None and width * height > self.max_pixels:
            scale = math.sqrt(self.max_pixels / (width * height))
            size = [max(1, math.floor(height * scale)), max(1, math.floor(width * scale))]
            image = F.resize(image, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        if self.training:
            if "wrist" not in camera_name.lower():
                width, height = image.size
                crop_size = (max(1, int(height * 0.95)), max(1, int(width * 0.95)))
                top, left, crop_h, crop_w = transforms.RandomCrop.get_params(image, crop_size)
                image = F.resized_crop(
                    image, top, left, crop_h, crop_w, [height, width],
                    interpolation=InterpolationMode.BILINEAR, antialias=True,
                )
                image = self.rotation(image)
            image = self.color(image)
        return F.to_tensor(image)
