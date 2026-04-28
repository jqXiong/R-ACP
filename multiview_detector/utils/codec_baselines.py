import io
import os
import shutil
import subprocess
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def normalized_tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    array = (array * IMAGENET_STD + IMAGENET_MEAN).clip(0.0, 1.0)
    return (array * 255.0 + 0.5).astype(np.uint8)


def uint8_image_to_normalized_tensor(image: np.ndarray) -> torch.Tensor:
    array = image.astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array).permute(2, 0, 1).float()


class TensorImageCodec:
    def __init__(
        self,
        codec: str,
        ffmpeg_bin: str = 'ffmpeg',
        jpeg_quality: int = 75,
        h264_crf: int = 28,
        h265_crf: int = 30,
        av1_crf: int = 35,
        use_cache: bool = True,
    ):
        self.codec = codec.lower()
        self.ffmpeg_bin = ffmpeg_bin
        self.jpeg_quality = jpeg_quality
        self.h264_crf = h264_crf
        self.h265_crf = h265_crf
        self.av1_crf = av1_crf
        self.use_cache = use_cache
        self._cache: Dict[str, Tuple[torch.Tensor, int]] = {}

    def encode_decode(self, tensor: torch.Tensor, cache_key: Optional[str] = None) -> Tuple[torch.Tensor, int]:
        if self.use_cache and cache_key is not None and cache_key in self._cache:
            decoded, encoded_bytes = self._cache[cache_key]
            return decoded.clone(), encoded_bytes

        image = normalized_tensor_to_uint8_image(tensor)
        if self.codec == 'jpeg':
            decoded, encoded_bytes = self._jpeg_roundtrip(image)
        elif self.codec in ('h264', 'h265', 'av1'):
            decoded, encoded_bytes = self._ffmpeg_roundtrip(image)
        else:
            raise ValueError(f'Unsupported codec baseline: {self.codec}')

        decoded_tensor = uint8_image_to_normalized_tensor(decoded)
        if self.use_cache and cache_key is not None:
            self._cache[cache_key] = (decoded_tensor.clone(), encoded_bytes)
        return decoded_tensor, encoded_bytes

    def _jpeg_roundtrip(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format='JPEG', quality=self.jpeg_quality)
        payload = buffer.getvalue()
        decoded = np.array(Image.open(io.BytesIO(payload)).convert('RGB'))
        return decoded, len(payload)

    def _ffmpeg_roundtrip(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        if shutil.which(self.ffmpeg_bin) is None:
            raise RuntimeError(
                f'ffmpeg binary "{self.ffmpeg_bin}" was not found. '
                'Install ffmpeg or pass --ffmpeg_bin with a valid path.'
            )

        suffix_map = {
            'h264': '.mp4',
            'h265': '.mp4',
            'av1': '.mkv',
        }
        with tempfile.TemporaryDirectory(prefix=f'codec_{self.codec}_') as tmpdir:
            input_path = os.path.join(tmpdir, 'frame.png')
            encoded_path = os.path.join(tmpdir, f'encoded{suffix_map[self.codec]}')
            decoded_path = os.path.join(tmpdir, 'decoded.png')
            Image.fromarray(image).save(input_path)

            if self.codec == 'h264':
                command = [
                    self.ffmpeg_bin,
                    '-y',
                    '-loglevel',
                    'error',
                    '-i',
                    input_path,
                    '-an',
                    '-c:v',
                    'libx264',
                    '-preset',
                    'veryfast',
                    '-tune',
                    'zerolatency',
                    '-g',
                    '1',
                    '-keyint_min',
                    '1',
                    '-crf',
                    str(self.h264_crf),
                    '-pix_fmt',
                    'yuv420p',
                    encoded_path,
                ]
            elif self.codec == 'h265':
                command = [
                    self.ffmpeg_bin,
                    '-y',
                    '-loglevel',
                    'error',
                    '-i',
                    input_path,
                    '-an',
                    '-c:v',
                    'libx265',
                    '-preset',
                    'fast',
                    '-x265-params',
                    f'keyint=1:min-keyint=1:no-scenecut=1:crf={self.h265_crf}',
                    '-pix_fmt',
                    'yuv420p',
                    encoded_path,
                ]
            else:
                command = [
                    self.ffmpeg_bin,
                    '-y',
                    '-loglevel',
                    'error',
                    '-i',
                    input_path,
                    '-an',
                    '-c:v',
                    'libaom-av1',
                    '-cpu-used',
                    '8',
                    '-crf',
                    str(self.av1_crf),
                    '-b:v',
                    '0',
                    '-still-picture',
                    '1',
                    '-pix_fmt',
                    'yuv420p',
                    encoded_path,
                ]
            subprocess.run(command, check=True)
            encoded_bytes = os.path.getsize(encoded_path)
            subprocess.run(
                [self.ffmpeg_bin, '-y', '-loglevel', 'error', '-i', encoded_path, decoded_path],
                check=True,
            )
            decoded = np.array(Image.open(decoded_path).convert('RGB'))
        return decoded, encoded_bytes


def apply_codec_packet_loss_to_batch(
    sequence_imgs: torch.Tensor,
    frame_ids,
    codec_runner: TensorImageCodec,
    packet_loss_rate: float,
    concealment: str = 'previous',
    transmitted_index: int = -1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.Tensor, float]:
    if rng is None:
        rng = np.random.default_rng(1)

    processed = sequence_imgs.clone()
    batch_size, time_steps, num_cam = processed.shape[:3]
    current_idx = transmitted_index if transmitted_index >= 0 else time_steps - 1
    total_kb = 0.0

    if isinstance(frame_ids, torch.Tensor):
        frame_key_list = frame_ids.detach().cpu().tolist()
    elif isinstance(frame_ids, (list, tuple)):
        frame_key_list = list(frame_ids)
    else:
        frame_key_list = [frame_ids]

    for b in range(batch_size):
        frame_key = frame_key_list[b]
        for cam in range(num_cam):
            cache_key = f'{codec_runner.codec}_{frame_key}_cam{cam}'
            decoded, encoded_bytes = codec_runner.encode_decode(processed[b, current_idx, cam], cache_key=cache_key)
            total_kb += encoded_bytes / 1024.0

            if rng.random() < packet_loss_rate:
                if concealment == 'previous' and current_idx > 0:
                    replacement = processed[b, current_idx - 1, cam].detach().cpu()
                else:
                    replacement = torch.zeros_like(decoded)
            else:
                replacement = decoded
            processed[b, current_idx, cam] = replacement.to(processed.device, dtype=processed.dtype)

    return processed, total_kb / max(batch_size, 1)
