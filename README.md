# Wan2.1 Text-to-Video Dev Container

This project provides a development container for working with the Wan2.1 text-to-video model, using components from the ComfyUI repackaged version.

## Setup Instructions

1. Make sure you have Docker and VS Code with the Dev Containers extension installed.

2. Clone this repository:
   ```bash
   git clone https://github.com/hollygrimm/module-wan2.1.git
   cd module-wan2.1
   ```

3. Open the project in VS Code and click "Reopen in Container" when prompted, or run the "Dev Containers: Reopen in Container" command from the Command Palette.

4. The container will build, downloading the model files from Hugging Face.

## Usage

### Using run_wan2.py

This script uses the model components downloaded from Hugging Face:

```bash
python run_wan2.py "Your prompt here"
```

Or with environment variables:

```bash
PROMPT="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window." python run_wan2.py
```

### Environment Variables

- `PROMPT`: The text prompt for video generation
- `NEGATIVE_PROMPT`: Text to avoid in generation (default: "")

## Troubleshooting

### Diffusers Version Issues

If you encounter errors related to the diffusers library:

1. We're using the latest version of diffusers at hash 6edb774b5e32f99987b89975b26f7c58b27ed111 which is known to work with Wan2.1
2. The model components are loaded directly from safetensors files to avoid compatibility issues

## License

Please note that the Wan2.1 model has its own license from Alibaba. Make sure to check their repository for usage restrictions.

## Acknowledgments

- Wan2.1 model from Alibaba
