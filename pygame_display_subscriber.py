#!/usr/bin/env python3

import asyncio
import logging
import pygame
import numpy as np
import sys
from faststream import FastStream
from faststream.redis import RedisBroker, PubSub
from faststream.redis.message import RedisMessage
from pydantic_settings import BaseSettings

from mirror_mirror.decode import decode_frame
from mirror_mirror.models import CarrierMessage, ProcessedFrameMessage

logger = logging.getLogger(__name__)


class Config(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    channel: str = "images:processed"
    window_name: str = "Mirror Mirror - Diffusion Output"
    show_fps: bool = True


config = Config()
broker = RedisBroker(url=config.redis_url)
app = FastStream(broker=broker)


class PygameDisplay:
    def __init__(self):
        self.screen = None
        self.clock = None
        self.running = True
        self.font = None
        self.fps_history = []
        self.max_fps_samples = 30
        
    def initialize(self, width: int = 800, height: int = 600):
        """Initialize pygame display"""
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(config.window_name)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.running = True
        logger.info(f"Initialized pygame display: {width}x{height}")
        
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                logger.info("Window close requested")
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    logger.info("ESC key pressed")
                    return False
                elif event.key == pygame.K_f:
                    config.show_fps = not config.show_fps
                    logger.info(f"FPS display: {'ON' if config.show_fps else 'OFF'}")
        return True
        
    def update_fps(self):
        """Update FPS calculation"""
        current_fps = self.clock.get_fps()
        self.fps_history.append(current_fps)
        if len(self.fps_history) > self.max_fps_samples:
            self.fps_history.pop(0)
            
    def get_average_fps(self) -> float:
        """Get average FPS over recent frames"""
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)
        
    def display_frame(self, rgb_array: np.ndarray, processing_time: float = 0.0):
        """Display RGB numpy array with optional performance overlay"""
        if not self.running or self.screen is None:
            return False
            
        # Handle events first
        if not self.handle_events():
            return False
            
        # Resize frame to fit screen if needed
        screen_size = self.screen.get_size()
        if rgb_array.shape[:2][::-1] != screen_size:
            # Convert to PIL Image for high-quality resizing
            import PIL.Image
            pil_image = PIL.Image.fromarray(rgb_array)
            pil_image = pil_image.resize(screen_size, PIL.Image.Resampling.LANCZOS)
            rgb_array = np.array(pil_image)
        
        # Convert numpy array to pygame surface
        surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
        
        # Blit the surface to screen
        self.screen.blit(surface, (0, 0))
        
        # Add performance overlay if enabled
        if config.show_fps:
            self.update_fps()
            avg_fps = self.get_average_fps()
            
            # Create text surfaces
            fps_text = self.font.render(f"FPS: {avg_fps:.1f}", True, (0, 255, 0))
            proc_text = self.font.render(f"Proc: {processing_time*1000:.1f}ms", True, (0, 255, 0))
            
            # Draw text with background for better visibility
            fps_rect = fps_text.get_rect()
            fps_rect.topleft = (10, 10)
            proc_rect = proc_text.get_rect()
            proc_rect.topleft = (10, 50)
            
            # Draw semi-transparent background
            s = pygame.Surface((max(fps_rect.width, proc_rect.width) + 20, 80))
            s.set_alpha(128)
            s.fill((0, 0, 0))
            self.screen.blit(s, (5, 5))
            
            # Draw text
            self.screen.blit(fps_text, fps_rect)
            self.screen.blit(proc_text, proc_rect)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)  # Limit to 60 FPS
        
        return True
        
    def cleanup(self):
        """Clean up pygame resources"""
        if pygame.get_init():
            pygame.quit()
        logger.info("Pygame cleaned up")


# Global display instance
display = PygameDisplay()


@broker.subscriber(channel=PubSub(config.channel))
async def display_processed_frames(
    carrier: CarrierMessage,
    _message: RedisMessage,
):
    """Display processed frames from diffusion pipeline"""
    
    if not isinstance(carrier.content, ProcessedFrameMessage):
        logger.warning(f"Unexpected message type: {type(carrier.content)}")
        return
    
    frame_msg = carrier.content
    
    try:
        # Decode frame from base64 JPEG to RGB numpy array
        rgb_array = decode_frame(frame_msg.frame)
        logger.debug(f"Decoded frame: {rgb_array.shape}, processing_time: {frame_msg.processing_time:.3f}s")
        
        # Initialize display on first frame
        if display.screen is None:
            height, width = rgb_array.shape[:2]
            display.initialize(width, height)
        
        # Display the frame
        success = display.display_frame(rgb_array, frame_msg.processing_time)
        
        # Check if display should close
        if not success or not display.running:
            logger.info("Display requested shutdown")
            display.cleanup()
            # Stop the FastStream app
            await app.stop()
            
    except Exception as e:
        logger.error(f"Error displaying frame: {e}")


@app.on_startup
async def startup_handler():
    """Handle app startup"""
    logger.info("Starting pygame display subscriber")
    logger.info(f"Subscribing to channel: {config.channel}")
    logger.info("Press ESC or close window to exit")
    logger.info("Press F to toggle FPS display")


@app.on_shutdown
async def shutdown_handler():
    """Handle app shutdown"""
    logger.info("Shutting down pygame display subscriber")
    display.cleanup()


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


if __name__ == "__main__":
    setup_logging()
    logger.info("Starting pygame display subscriber...")
    
    try:
        # Run the FastStream app
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error running app: {e}")
    finally:
        display.cleanup()
        logger.info("Pygame display subscriber stopped") 