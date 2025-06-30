import logging
import time
from functools import cache

import numpy as np
import torch
from diffusers import StableDiffusionPipeline, AutoencoderTiny
from faststream import Depends, FastStream
from faststream.redis import PubSub, RedisBroker
from PIL import Image, ImageOps
from pydantic_settings import BaseSettings

from mirror_mirror.common import log_errors
from mirror_mirror.decode import decode_frame
from mirror_mirror.models import CarrierMessage, FrameMessage, LatentsMessage, decode_bytes, serialize_array

logger = logging.getLogger(__name__)


class Config(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    model_repo: str = "IDKiro/sdxs-512-dreamshaper"
    torch_dtype: str = "float16"
    device: str = "cuda"
    target_size: int = 512


config = Config()
broker = RedisBroker(url=config.redis_url)
app = FastStream(broker=broker)


def np_to_pt(image_array):
    return torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)


class LatentEncoder:
    def __init__(self):
        self.vae = None
        self.device = config.device
        self.dtype = getattr(torch, config.torch_dtype)
        self.frames_processed = 0
        self.total_processing_time = 0.0
        logger.info(f"Loading VAE from {config.model_repo}")
        start_time = time.time()

        pipe = StableDiffusionPipeline.from_pretrained(
            config.model_repo,
            torch_dtype=self.dtype,
        )
        self.vae: AutoencoderTiny = pipe.vae
        self.vae.to(self.device)
        self.vae.eval()

        load_time = time.time() - start_time
        logger.info(f"VAE loaded successfully in {load_time:.2f}s - device: {self.device}, dtype: {self.dtype}")

    @torch.inference_mode()
    def encode_frame(self, frame_bytes: bytes) -> tuple[bytes, tuple[int, ...], str]:
        """Encode a frame to latents"""
        # Decode frame from bytes
        frame = decode_frame(frame_bytes)

        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        pil_image = Image.fromarray(frame)

        # Resize to target size
        pil_image = ImageOps.fit(pil_image, (config.target_size, config.target_size))

        # Convert to tensor and normalize
        image_array = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = np_to_pt(image_array)
        image_tensor = image_tensor.to(self.device, dtype=self.dtype)

        # Encode to latents
        output = self.vae.encode(image_tensor)

        # Handle different VAE output formats
        if hasattr(output, "latent_dist"):
            # AutoencoderKL format
            latents = output.latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        else:
            # AutoencoderTiny format - returns latents directly
            latents = output.latents
            # AutoencoderTiny doesn't need scaling

        # Convert to numpy and serialize
        latents_np = latents.cpu().float().numpy()

        return serialize_array(latents_np)


@cache
def get_encoder() -> LatentEncoder:
    return LatentEncoder()


@broker.subscriber(channel=PubSub("frames:camera:*", pattern=True))
@broker.publisher(channel="latents:camera")
@log_errors
async def encode_camera_frames(
    carrier: CarrierMessage,
    encoder: LatentEncoder = Depends(get_encoder),
) -> CarrierMessage | None:
    """Process camera frames and convert to latents"""

    if not isinstance(carrier.content, FrameMessage):
        logger.debug(f"Ignoring non-frame message: {type(carrier.content)}")
        return None

    frame_msg = carrier.content

    # Decode and validate frame data
    frame_bytes = decode_bytes(frame_msg.frame)

    # Encode frame to latents
    latents_data, shape, dtype = encoder.encode_frame(frame_bytes)

    latents_msg = LatentsMessage(latents=latents_data, shape=shape, dtype=dtype, source="camera")

    return CarrierMessage(content=latents_msg)


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting latent encoder...")
    asyncio.run(app.run())
