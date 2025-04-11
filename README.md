# Wan2.1 Text-to-Video Module for Lilypad

This project provides a module for running the Wan2.1 text-to-video model on the Lilypad network. The module generates WebP video files at 832x480 resolution with 49 frames at 16 FPS.

## Prerequisites

- [Lilypad CLI installed](https://docs.lilypad.tech/lilypad/lilypad-testnet/install-run-requirements)
- Docker installed on your system (for local development)
- GPU with at least 1 GPU
- At least 24GB RAM

## Running on Lilypad Network

### Using Lilypad LocalNet

To run on the local development network:

```
go run . run --network dev github.com/hollygrimm/module-wan2.1:<COMMIT_HASH> --web3-private-key <JOB_CREATOR_PRIVATE_KEY> -i prompt="your prompt here" -i negative_prompt="your negative prompt here"
```

Replace `<COMMIT_HASH>` with the latest commit hash from this repository and `<JOB_CREATOR_PRIVATE_KEY>` with the Job Creator Private Key found in the lilypad repository file `.local.dev`.

### Using Lilypad Main Network

To run on the main Lilypad network:

```
lilypad run github.com/hollygrimm/module-wan2.1:<COMMIT_HASH> -i prompt="your prompt here" -i negative_prompt="your negative prompt here"
```

## Development Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/hollygrimm/module-wan2.1.git
   cd module-wan2.1
   ```

2. If using VS Code with Dev Containers, open the project and click "Reopen in Container" when prompted, or run the "Dev Containers: Reopen in Container" command from the Command Palette.

3. The container will build, downloading the model files from Hugging Face.

### Local Development with run_wan2.1.py

```bash
python run_wan2.1.py "Your prompt here"
```

Or with environment variables:

```bash
PROMPT="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window." python run_wan2.1.py
```

Or using Docker directly:

```bash
mkdir -p ./outputs && docker run --gpus all -v $(pwd)/outputs:/outputs -e "PROMPT=Two frogs sit on a lilypad, animatedly discussing the wonders and quirks of AI agents. As they ponder whether these digital beings can truly understand their froggy lives, the serene pond serves as a backdrop to their lively conversation." --rm hollygrimm/wan2.1-text2video-ipfs:latest
```

### Environment Variables

- `PROMPT`: The text prompt for video generation
- `NEGATIVE_PROMPT`: Text to avoid in generation (default: "")
  
  Example negative prompt:
  ```
  Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
  ```

### Push to Docker Hub

```bash
docker build -t hollygrimm/wan2.1-text2video-ipfs:latest --target production .
docker push hollygrimm/wan2.1-text2video-ipfs:latest
```

## Notes

- The module requires at least 24GB of RAM and 1 GPU
- Video generation takes approximately 3-5 minutes on an RTX 4090
- Ensure you have the necessary permissions and resources to run Docker containers with GPU support

## Troubleshooting

### Diffusers Version Issues

If you encounter errors related to the diffusers library:

1. We're using the latest version of [diffusers](https://github.com/huggingface/diffusers) at hash `6edb774b5e32f99987b89975b26f7c58b27ed111` which includes `AutoencoderKLWan` for Wan2.1
2. The model components are loaded directly from safetensors files to avoid compatibility issues

## License

Please note that the Wan2.1 model has its own license from Alibaba. Make sure to check their repository for usage restrictions.

## Acknowledgments

- [Wan2.1 model from Alibaba/Wan-Video](https://github.com/Wan-Video/Wan2.1)
- [Diffusers library](https://github.com/huggingface/diffusers)
