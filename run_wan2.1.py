import sys
import torch
import os
import logging
import gc
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_prompt():
    # Use sys.argv[1] if available, otherwise fall back to environment variable or default
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])  # Join all arguments to allow for prompts with spaces
    else:
        prompt = os.environ.get('PROMPT', os.environ.get('DEFAULT_PROMPT', "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."))
    logging.info(f"Using prompt: {prompt}")
    return prompt

def clear_cuda_cache():
    """Clear CUDA cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def main():
    try:
        # Configure CUDA for better memory management
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 precision for better performance
        torch.backends.cudnn.benchmark = True  # Optimize CUDA operations
        
        # Clear CUDA cache to start fresh
        clear_cuda_cache()
        
        logging.info("Starting Wan2.1 text-to-video generation")
        
        # Get the prompt
        prompt = get_prompt()
        # negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        negative_prompt = os.environ.get('NEGATIVE_PROMPT', '') 
        logging.info(f"Negative prompt: {negative_prompt}")

        # Set default values directly
        resolution = '832*480'
        steps = 30
        guide_scale = 6.0
        seed = -1
        flow_shift = 3.0  # 3.0 for 480p recommended
        use_attention_slicing = True
        
        # Parse resolution
        W = int(resolution.split("*")[0])
        H = int(resolution.split("*")[1])
        
        # Log resolution in correct order for clarity
        logging.info(f"Using resolution: {W}x{H}")
        
        # Import diffusers modules
        from diffusers import AutoencoderKLWan, WanPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from diffusers.utils import export_to_video
        
        # Set up random seed for reproducibility
        if seed >= 0:
            torch.manual_seed(seed)
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None
            
        # Initialize text to video model with diffusers
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        
        # Load VAE with float32 precision for better quality
        logging.info(f"Loading VAE from {model_id}")
        vae = AutoencoderKLWan.from_pretrained(
            model_id, 
            subfolder="vae", 
            torch_dtype=torch.float32
        )
        
        # Load the pipeline with float16 precision for the other components
        logging.info(f"Loading pipeline from {model_id}")
        pipe = WanPipeline.from_pretrained(
            model_id, 
            vae=vae, 
            torch_dtype=torch.float16
        )
        
        # Enable automatic CPU offloading
        logging.info("Enabling automatic model CPU offloading")
        pipe.enable_model_cpu_offload()
        
        if use_attention_slicing:
            logging.info("Enabling attention slicing")
            pipe.enable_attention_slicing(1)
        
        # Configure scheduler with flow_shift
        pipe.scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True, 
            num_train_timesteps=1000, 
            flow_shift=flow_shift
        )
        
        # Generate latents
        try:
            logging.info(f"Starting generation with resolution {W}x{H}, {steps} steps")
            video_frames = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=H,
                width=W,
                num_frames=49,  # Default number of frames
                guidance_scale=guide_scale,
                generator=generator,
                num_inference_steps=steps
            ).frames[0]
            
            # Clean up memory
            clear_cuda_cache()
            
            # Save the video in the outputs directory
            output_dir = "/outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as WebP format too
            webp_output_path = os.path.join(output_dir, "output.webp")
            logging.info(f"Saving WebP video to {webp_output_path}")
            export_to_video(video_frames, webp_output_path, fps=16)
            
            logging.info(f"Video generated and saved as {webp_output_path}")
            
        except RuntimeError as e:
            logging.error(f"CUDA error during generation: {str(e)}")
            logging.info("Try reducing resolution, number of frames, or inference steps")
            raise e
            
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 