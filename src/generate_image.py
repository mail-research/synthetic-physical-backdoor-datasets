from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionInstructPix2PixPipeline
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
import torch

from argparse import ArgumentParser

import os
import numpy as np
import matplotlib.pyplot as plt

import random

from PIL import Image
import uuid

def parse_args():
    parser = ArgumentParser('Generation script with Stable Diffusion')

    parser.add_argument('--generation_method', type=str, default='stable_diffusion', choices=['stable_diffusion', 'instruct_pix2pix'])
    parser.add_argument('--model_path', type=str, default='stabilityai/stable-diffusion-2-1-base')

    parser.add_argument('--dtype', type=str, choices=['float16', 'float32', 'float64'], default='float16')
    parser.add_argument('--enable_xformers', action='store_true')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--negative_prompt', type=str, required=False, default='')
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=10)

    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)

    parser.add_argument('--num_samples', type=int, default=20)

    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument('--random_prompt', default=None, type=str)
    parser.add_argument('--background', default=None, type=str)
    parser.add_argument('--action', default=None, type=str)

    # Instruct-Pix2Pix params
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--image_guidance_scale', type=float, default=1.5)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = args.model_path

    if args.dtype == 'float16':
        dtype = torch.float16
    elif args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float64':
        dtype = torch.float64
    
    if args.generation_method == 'stable_diffusion':
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype
        )
        


        if args.enable_xformers:
            pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            # Workaround for not accepting attention shape using VAE for Flash Attention
            pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)

        vae = pipe.vae.to(device)
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder.to(device)
        unet = pipe.unet.to(device)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        remainder = args.num_samples % 20
        num_iter = args.num_samples // 20 if remainder == 0 else args.num_samples // 20 + 1
        for j in range(num_iter):
            if (remainder != 0) and (j == num_iter - 1):
                num_images = remainder
            else:
                num_images = 20

            prompt = args.prompt

            if args.random_prompt is not None:
                # args.random_prompt += ', '
                rand_prompt = args.random_prompt.split(',')
                rand = random.choice(rand_prompt)
                prompt += f', {rand}'
            if args.background is not None:
                bg = args.background.split(",")
                bg = random.choice(bg)
                prompt += f', ({bg})'
            if args.action is not None:
                act = args.action.split(",")
                act = random.choice(act)
                prompt += f", {act}"
            
            images = pipe(
                prompt=prompt, # Combine with multiple scenario like running, sitting, lying, etc
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, # Using multiple guidance scale
                negative_prompt=args.negative_prompt,
                num_images_per_prompt=num_images,
            ).images

            if args.output_dir is None:
                args.output_dir = './generated_outputs'
                output_dir = os.path.join(args.output_dir, '_'.join(args.prompt.split()))
            else:
                output_dir = args.output_dir

            os.makedirs(output_dir, exist_ok=True)

            for i in range(len(images)):
                # images[i].save(f'{output_dir}/image_{(j*len(images))+i}.png')
                images[i].save(f'{output_dir}/{str(uuid.uuid4())}.png')

            # ims = []
            # for i in range(len(images) // 4):
            #     ims.append(np.concatenate(images[i * 4: (i+1) * 4], axis=1))
            # fig, ax = plt.subplots(4, 5, figsize=(30, 20))
            # for a in range(4):
            #     for b in range(5):
            #         ax[a][b].imshow(images[(a * 5) + b])
            #         ax[a][b].axis('off')
            #         ax[a][b].set_title(f'image_{(num_images * (a*5)) + b}')

            # # ims = np.concatenate(ims, axis=0)
            # # fig, ax = plt.subplots(1, 1, figsize=(20, 30))
            # # ax.imshow(ims)
            # # ax.axis('off')
            # # plt.show()
            # plt.tight_layout()
            # plt.savefig(f'{args.output_dir}/combined_{j}.png')
            # plt.close()
    elif args.generation_method == 'instruct_pix2pix':
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=dtype
        )
        pipe = pipe.to(device)
        
        vae = pipe.vae.to(device)
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder.to(device)
        unet = pipe.unet.to(device)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        # remainder = args.num_samples % 20
        # num_iter = args.num_samples // 20 if remainder == 0 else args.num_samples // 20 + 1
        sample_path = os.listdir(args.image_dir)
        # print(len(sample_path))
        if len(sample_path) < args.num_samples:
            raise Exception(f'Number of samples is greater than number of source images')

        for j in range(args.num_samples):
            # if (remainder != 0) and (j == num_iter - 1):
            #     num_images = remainder
            # else:
            #     num_images = 20
            # image = 

            images = pipe(
                image=Image.open(os.path.join(args.image_dir, sample_path[j])),
                prompt=args.prompt,
                # height=args.height,
                # width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, 
                image_guidance_scale=args.image_guidance_scale,
                negative_prompt=args.negative_prompt,
                # num_images_per_prompt=num_images,
            ).images

            output_dir = args.output_dir

            os.makedirs(output_dir, exist_ok=True)

            for i in range(len(images)):
                images[i].save(f'{output_dir}/image_{(j*len(images))+i}.png')

            # ims = []
            # for i in range(len(images) // 4):
            #     ims.append(np.concatenate(images[i * 4: (i+1) * 4], axis=1))
            # fig, ax = plt.subplots(4, 5, figsize=(30, 20))
            # for a in range(4):
            #     for b in range(5):
            #         ax[a][b].imshow(images[(a * 5) + b])
            #         ax[a][b].axis('off')
            #         ax[a][b].set_title(f'image_{(num_images * (a*5)) + b}')

            # # ims = np.concatenate(ims, axis=0)
            # # fig, ax = plt.subplots(1, 1, figsize=(20, 30))
            # # ax.imshow(ims)
            # # ax.axis('off')
            # # plt.show()
            # plt.tight_layout()
            # plt.savefig(f'{args.output_dir}/combined_{j}.png')
            # plt.close()
