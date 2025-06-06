# Mirror Mirror

A real-time camera-based image diffusion system for Jetson Orin NX.

## Architecture

The system follows a modular pipeline architecture using FastStream and Redis for message passing:

```
Camera → Preprocess → Latent Encoder → Diffusion → Latent Decoder → Display
                                            ↑
Mic → VAD → STT → Rephrase -> Tokenize → Embed
```

### Components

1. **Camera Server** (`camera_server.py`) - Captures webcam frames
2. **Latent Encoder** (`latent_encoder.py`) - Converts frames to latent space using VAE
3. **Diffusion Server** (`diffusion_server.py`) - Applies diffusion effects using SDXS
4. **Latent Decoder** (`latent_decoder.py`) - Converts latents back to images
5. **Display** (`display.py`) - Shows processed frames with performance metrics
6. **Audio Server** (`audio_server.py`) - Voice activity detection and audio capture
7. **Prompt Publisher** (`prompt_publisher.py`) - Publishes text prompts

### Message Flow

- **Redis Channels**:
  - `frames:camera:{id}` - Raw camera frames
  - `latents:camera` - Encoded camera latents
  - `latents:diffused` - Processed latents from diffusion
  - `images:processed` - Final processed images
  - `prompts:*` - Text prompts for diffusion
  - `audio:chunks` - Audio data from microphone

# Architecture:

Local Redis Broker: (add a docker compose to run it on the device)
Using FastStream for piping messages between components
Each component in its own small python script, sharing api shapes with a common imported module
The Diffusion module accepts both embedded latents and prompts
The Prompts should come from a VAD -> STT -> Rephrading (English to image genration descriptions) -> Tokenzation -> Embedding 
Images should come from Camera (cv2) -> preprocessing (crop resize to_tensor) -> ControlNet -> Encoding To Latent
Diffusion subscribes to both Image and Prompt streams (memoizes the prompt) -> Diffusion -> Decodeing -> Denormalization -> Postprocessing (resize, crop, to_tensor) 
Display module subscribes to the processed images and displays to the screen

