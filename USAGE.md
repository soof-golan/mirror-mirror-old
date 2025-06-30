# Pygame Display Subscriber Usage

## Overview

The `pygame_display_subscriber.py` script subscribes to the mirror-mirror diffusion pipeline and displays the processed images in real-time using pygame.

## Features

- **Real-time display**: Shows images as they are processed by the diffusion pipeline
- **Performance monitoring**: Displays FPS and processing time metrics
- **Interactive controls**: 
  - ESC key or window close to exit
  - F key to toggle FPS display
  - F11 key to toggle fullscreen/windowed mode
- **Auto-scaling**: Automatically resizes images to fit the window
- **High-quality display**: Uses PIL for image resizing to maintain quality

## Installation

The subscriber is included in the mirror-mirror project. All dependencies are already specified in `pyproject.toml`.

## Usage

### Method 1: Direct execution
```bash
uv run --no-project python pygame_display_subscriber.py
```

### Method 2: Using the script entry point
```bash
uv run pygame-display
```

## Configuration

The subscriber can be configured using environment variables:

- `REDIS_URL`: Redis connection URL (default: `redis://localhost:6379`)
- `CHANNEL`: Redis channel to subscribe to (default: `frames:camera:0`)
- `WINDOW_NAME`: Window title (default: `Mirror Mirror - Diffusion Output`)
- `SHOW_FPS`: Show FPS overlay (default: `True`)
- `FULLSCREEN`: Start in fullscreen mode (default: `True`)

Example:
```bash
# Run with custom Redis URL and disable FPS display
REDIS_URL=redis://localhost:6380 SHOW_FPS=False uv run python pygame_display_subscriber.py

# Run in windowed mode instead of fullscreen
FULLSCREEN=False uv run python pygame_display_subscriber.py
```

## Integration with Mirror-Mirror Pipeline

The subscriber is designed to work with the complete mirror-mirror pipeline:

1. **Camera Server** → captures frames → publishes to `frames:camera:*`
2. **Latent Encoder** → encodes frames to latents → publishes to `latents:camera`
3. **Diffusion Server** → processes latents → publishes to `latents:diffused`
4. **Latent Decoder** → decodes latents to images → publishes to `images:processed`
5. **Pygame Display Subscriber** → **displays processed images** ← subscribes to `images:processed`

## Running the Full Pipeline

To test the subscriber with the full pipeline:

```bash
# Start Redis (if not already running)
redis-server

# In separate terminals, start each component:

# 1. Camera server
uv run python -m mirror_mirror.camera_server

# 2. Latent encoder  
uv run python -m mirror_mirror.latent_encoder

# 3. Diffusion server
uv run python -m mirror_mirror.diffusion_server

# 4. Latent decoder
uv run python -m mirror_mirror.latent_decoder

# 5. Pygame display subscriber
uv run python pygame_display_subscriber.py
```

## Testing

Test the subscriber without the full pipeline:

```bash
uv run --no-project python test_pygame_subscriber.py
```

This will verify:
- Pygame can initialize properly
- Redis connection works
- All imports are successful

## Troubleshooting

### Common Issues

1. **Redis connection error**: Make sure Redis is running on the configured URL
2. **Pygame initialization error**: Make sure you have a display available (not running headless)
3. **Import errors**: Use `uv run --no-project` to avoid build issues

### Logs

The subscriber provides detailed logging. Increase log level for debugging:

```python
# In pygame_display_subscriber.py, change:
logging.basicConfig(level=logging.DEBUG)  # Instead of INFO
```

## Standalone RGB Array Display

For displaying RGB numpy arrays outside the pipeline, use the standalone script:

```bash
uv run --no-project python display_rgb_array.py
```

This creates a sample gradient and displays it, useful for testing pygame functionality independently. 