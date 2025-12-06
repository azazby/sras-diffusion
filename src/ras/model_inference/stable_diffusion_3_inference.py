import argparse
import torch
from diffusers import StableDiffusion3Pipeline
from ras.utils.stable_diffusion_3.update_pipeline_sd3 import update_sd3_pipeline
from ras.utils import ras_manager
from ras.utils.ras_argparser import parse_args
import csv
import os

def sd3_inf(args):
    pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", dtype=torch.float16, use_auth_token=True)
    pipeline.to("cuda")
    pipeline = update_sd3_pipeline(pipeline)
    numsteps = args.num_inference_steps

    # Parse Prompts
    # we expect prompt_file to be a CSV where: Col 1 = Prompt, Col 2 = Index
    prompts_data = []
    # If given prompt file
    if hasattr(args, 'prompt_file') and args.prompt_file:
        print(f"Reading prompts from {args.prompt_file}...")
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|') # expect '|' delimiter
            for row in reader:
                if len(row) >= 2:
                    prompts_data.append((row[0].strip(), row[1].strip()))
    else:
        # Fallback to single CLI prompt if no file provided
        prompts_data.append((args.prompt, "0"))

    # Check Output File
    output_dir = args.output
    # If we are given prompt file, check if output directory exists
    # If the user passed a filename like 'out.png', we strip it to get the folder
    if output_dir.endswith('.png'):
        output_dir = os.path.dirname(output_dir)
    # Make output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # Inference Loop
    for i, prompt_data in enumerate(prompts_data):
        prompt_text, file_idx = prompt_data
        print(f"Generating Img {i+1}/{len(prompts_data)}: Index {file_idx}: {prompt_text[:40]}...")
        # reset RAS Manager state
        ras_manager.MANAGER.reset_internal_state()
        # reset Generator seed each loop
        generator = torch.Generator("cuda").manual_seed(args.seed) if args.seed is not None else None
        image = pipeline(
                        generator=generator,
                        num_inference_steps=numsteps,
                        prompt=prompt_text,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        guidance_scale=7.0,
                        ).images[0]
        # Save to Drive
        # Save Image
        img_filename = f"img_{file_idx}.png"
        full_img_path = os.path.join(output_dir, img_filename)
        image.save(full_img_path)
        print(f"Saved Img to Drive: {full_img_path}")
        # Save Attention Maps if desired for visualization testing
        if args.save_attn:
            attn_filename = f"attn_scores_{file_idx}.pt"
            full_attn_path = os.path.join(output_dir, attn_filename)
            torch.save(ras_manager.MANAGER.attn_scores, full_attn_path)
            print(f"Saved Attn Map to Drive: {full_img_path}")

if __name__ == "__main__":
    args = parse_args()
    ras_manager.MANAGER.set_parameters(args)
    sd3_inf(args)