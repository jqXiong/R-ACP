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
        h264_encoder: Optional[str] = None,
        h265_encoder: Optional[str] = None,
        av1_encoder: Optional[str] = None,
        use_cache: bool = True,
    ):
        self.codec = codec.lower()
        self.ffmpeg_bin = ffmpeg_bin
        self.jpeg_quality = jpeg_quality
        self.h264_crf = h264_crf
        self.h265_crf = h265_crf
        self.av1_crf = av1_crf
        self.h264_encoder = h264_encoder
        self.h265_encoder = h265_encoder
        self.av1_encoder = av1_encoder
        self.use_cache = use_cache
        self._cache: Dict[str, Tuple[torch.Tensor, int]] = {}
        self._available_encoders = None

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

            command_candidates = self._build_encoder_commands(input_path, encoded_path)
            self._run_ffmpeg_candidates(command_candidates)
            encoded_bytes = os.path.getsize(encoded_path)
            self._run_ffmpeg_candidates([
                [self.ffmpeg_bin, '-y', '-loglevel', 'error', '-i', encoded_path, decoded_path]
            ])
            decoded = np.array(Image.open(decoded_path).convert('RGB'))
        return decoded, encoded_bytes

    def _get_available_encoders(self):
        if self._available_encoders is not None:
            return self._available_encoders
        result = subprocess.run(
            [self.ffmpeg_bin, '-hide_banner', '-encoders'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        encoders = set()
        for line in (result.stdout or '').splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith('V'):
                encoders.add(parts[1])
        self._available_encoders = encoders
        return encoders

    def _preferred_encoders(self):
        if self.codec == 'h264':
            override = [self.h264_encoder] if self.h264_encoder else []
            preferred = ['libx264', 'libopenh264', 'h264_nvenc', 'h264_qsv', 'h264_vaapi', 'h264_v4l2m2m', 'h264_amf', 'h264_mf']
        elif self.codec == 'h265':
            override = [self.h265_encoder] if self.h265_encoder else []
            preferred = ['libx265', 'hevc_nvenc', 'hevc_qsv', 'hevc_vaapi', 'hevc_v4l2m2m', 'hevc_amf', 'hevc_mf']
        else:
            override = [self.av1_encoder] if self.av1_encoder else []
            preferred = ['libaom-av1', 'svtav1', 'rav1e', 'av1_nvenc', 'av1_qsv', 'av1_amf', 'av1_vaapi']
        available = self._get_available_encoders()
        ordered = []
        for name in override + preferred:
            if name and name in available and name not in ordered:
                ordered.append(name)
        return ordered

    def _build_encoder_commands(self, input_path: str, encoded_path: str):
        encoder_names = self._preferred_encoders()
        if not encoder_names:
            raise RuntimeError(
                f'No usable ffmpeg encoder found for codec "{self.codec}". '
                f'You can pass an explicit encoder override for this codec.'
            )
        commands = []
        for encoder in encoder_names:
            command = [
                self.ffmpeg_bin,
                '-y',
                '-loglevel',
                'error',
                '-i',
                input_path,
                '-frames:v',
                '1',
                '-an',
                '-c:v',
                encoder,
            ]
            if encoder == 'libx264':
                command += ['-crf', str(self.h264_crf), '-pix_fmt', 'yuv420p']
            elif encoder == 'libx265':
                command += ['-crf', str(self.h265_crf), '-pix_fmt', 'yuv420p']
            elif encoder == 'libaom-av1':
                command += ['-crf', str(self.av1_crf), '-b:v', '0', '-still-picture', '1', '-pix_fmt', 'yuv420p']
            elif encoder in ('svtav1', 'rav1e'):
                command += ['-crf', str(self.av1_crf), '-pix_fmt', 'yuv420p']
            else:
                command += ['-pix_fmt', 'yuv420p']
            command.append(encoded_path)
            commands.append(command)
        return commands

    def _run_ffmpeg_candidates(self, command_candidates):
        errors = []
        for command in command_candidates:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                return
            errors.append(
                {
                    'command': command,
                    'stderr': (result.stderr or '').strip(),
                    'stdout': (result.stdout or '').strip(),
                    'returncode': result.returncode,
                }
            )
        message_lines = ['All ffmpeg command candidates failed.']
        for idx, item in enumerate(errors, start=1):
            message_lines.append(f'Candidate {idx} rc={item["returncode"]}: {" ".join(item["command"])}')
            if item['stderr']:
                message_lines.append(f'stderr: {item["stderr"]}')
            if item['stdout']:
                message_lines.append(f'stdout: {item["stdout"]}')
        raise RuntimeError('\n'.join(message_lines))


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
