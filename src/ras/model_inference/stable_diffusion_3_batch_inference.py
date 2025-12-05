import argparse
import torch
from diffusers import StableDiffusion3Pipeline
from ras.utils.stable_diffusion_3.update_pipeline_sd3 import update_sd3_pipeline
from ras.utils import ras_manager
from ras.utils.ras_argparser import parse_args
import os

def sd3_inf_batch(args):# Load pipeline once
    pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, use_auth_token=True)
    pipeline.to("cuda")
    pipeline = update_sd3_pipeline(pipeline)
    
    with open(args.prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    # Create the output directory specified by the args
    os.makedirs(args.output_dir, exist_ok=True)
    
    for idx, prompt in enumerate(prompts):
        print(f"Processing {idx+1}/{len(prompts)}: {prompt[:50]}...")
        generator = torch.Generator("cuda").manual_seed(args.seed + idx) if args.seed is not None else None
        
        image = pipeline(
            generator=generator,
            num_inference_steps=args.num_inference_steps,
            prompt=prompt,
            negative_prompt=args.negative_prompt if hasattr(args, 'negative_prompt') else None,
            height=args.height if hasattr(args, 'height') else 512,
            width=args.width if hasattr(args, 'width') else 512,
            guidance_scale=7.0,
        ).images[0]
        image.save(f"{args.output_dir}/image_{idx:04d}.png")
        
        # Save metadata
        with open(f"{args.output_dir}/metadata_{idx:04d}.txt", 'w') as f:
            f.write(f"Prompt: {prompt}\nSeed: {args.seed + idx if args.seed else 'None'}\n")

if __name__ == "__main__":
    args = parse_args()
    ras_manager.MANAGER.set_parameters(args)
    sd3_inf_batch(args)