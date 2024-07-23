# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

from __future__ import annotations


import os
import math
import random
import sys
from argparse import ArgumentParser

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

import requests
import importlib

sys.path.append("/vinserver_user/jason/diffusion_bd/InstructDiffusion/stable_diffusion")


class EditDataset(Dataset):
    def __init__(self, input_folder:str, resolution:int) -> None:
        super().__init__()
        # list all image inside the image folder
        self.list_img = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
        self.transforms = transforms.ToTensor()
        self.resolution = resolution

    def __len__(self):
        return len(self.list_img)
    
    def __getitem__(self, index):
        # get image name
        img_path = self.list_img[index]
        file_name = img_path.split('/')[-1]     # remove the folder path
        file_name = file_name.split('.')[0]     # remove the extension
        # load the image
        input_image = Image.open(img_path).convert("RGB")
        width, height = input_image.size
        factor = self.resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width_resize = int((width * factor) // 64) * 64
        height_resize = int((height * factor) // 64) * 64
        # input_image = ImageOps.fit(input_image, (width_resize, height_resize), method=Image.Resampling.LANCZOS)
        input_image = transforms.Resize(self.resolution)(input_image)
        input_image = transforms.CenterCrop(self.resolution)(input_image)
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> c h w")
        return input_image, file_name, width, height



# from stable_diffusion.ldm.util import instantiate_from_config
def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "b ... -> (repeat b) ...", repeat=3)
        # cfg_z = z
        cfg_sigma = einops.repeat(sigma, "b ... -> (repeat b) ...", repeat=3)
        # cfg_sigma = sigma
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], cond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_txt_cond \
            = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
            
        return 0.5 * (out_img_cond + out_txt_cond) + \
            text_cfg_scale * (out_cond - out_img_cond) + \
                image_cfg_scale * (out_cond - out_txt_cond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    model = instantiate_from_config(config.model)

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if 'state_dict' in pl_sd:
        pl_sd = pl_sd['state_dict']
    m, u = model.load_state_dict(pl_sd, strict=False)

    print(m, u)
    return model

def resize_img(input_image, args):
    width, height = input_image.size
    factor = args.resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width_resize = int((width * factor) // 64) * 64
    height_resize = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width_resize, height_resize), method=Image.Resampling.LANCZOS)
    
    return input_image


def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="../InstructDiffusion/configs/instruct_diffusion.yaml", type=str)
    parser.add_argument("--ckpt", default="../InstructDiffusion/configs/v1-5-pruned-emaonly-adaption-task.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    # parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--outdir", default="../edited_outputs", type=str)
    parser.add_argument("--edit", required=True, type=str)
    parser.add_argument("--cfg-text", default=5.0, type=float)
    parser.add_argument("--cfg-image", default=1.25, type=float)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument('--batch_size', type=int, default=4) # EDIT
    
    parser.add_argument("--edit-rate", type=float, default=0.05)
    
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""] * args.batch_size)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    random.seed(0 if args.seed is None else args.seed)
    img_files = os.listdir(args.input_dir)
    img_files = random.choices(img_files, k=int(args.edit_rate * len(img_files)))
    img_files = [os.path.join(args.input_dir, i) for i in img_files]
    
    #############CHANGE HERE#################
    dataset = EditDataset(input_folder=args.input_dir, resolution=args.resolution)
    idx = torch.arange(0, len(dataset))
    idx = random.choices(idx, k=int(args.edit_rate * len(dataset)))
    dataset = torch.utils.data.Subset(dataset, idx)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=False)
    output_dir = args.outdir
    for idx, data in enumerate(data_loader):
        imgs, fnames, widths, heights = data
        seed = random.randint(0, 100000) if args.seed is None else args.seed
        with torch.no_grad(), autocast("cuda"):
            cond = {}
            cond["c_crossattn"] = [model.get_learned_conditioning([args.edit] * args.batch_size)]
            imgs = imgs.to(next(model.parameters()).device)
            cond["c_concat"] = [model.encode_first_stage(imgs).mode()]
            # print(imgs.shape)
            # print("conditional")
            # print(len(cond["c_concat"]))
            # print(cond["c_concat"][0].shape)
            # print(cond["c_crossattn"][0].shape)
            uncond = {}
            uncond["c_crossattn"] = [null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]
            # print("unconditional")
            # print(uncond["c_crossattn"][0].shape)
            # print(uncond["c_concat"][0].shape)
            sigmas = model_wrap.get_sigmas(args.steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond, 
                "text_cfg_scale": args.cfg_text,
                "image_cfg_scale": args.cfg_image,
            }

            torch.manual_seed(seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "b c h w -> b h w c")

        for i in range(len(fnames)):
            
            edited_image = Image.fromarray(x[i].type(torch.uint8).cpu().numpy())

            edited_image = ImageOps.fit(edited_image, (widths[i], heights[i]), method=Image.Resampling.LANCZOS)
            edited_image.save(output_dir+'/output_'+fnames[i]+'_seed'+str(seed)+'.png')

    ############END##########################
    # batch = 4
    # steps = math.ceil(len(img_files) / batch)
    # for input in img_files:
    #     seed = random.randint(0, 100000) if args.seed is None else args.seed
    #     # continue
    #     input_image = Image.open(input).convert("RGB")
    #     width, height = input_image.size
    #     factor = args.resolution / max(width, height)
    #     factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    #     width_resize = int((width * factor) // 64) * 64
    #     height_resize = int((height * factor) // 64) * 64
    #     input_image = ImageOps.fit(input_image, (width_resize, height_resize), method=Image.Resampling.LANCZOS)

    #     output_dir = args.outdir
    #     os.makedirs(output_dir, exist_ok=True)
    #     with torch.no_grad(), autocast("cuda"):
    #         cond = {}
    #         cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
    #         input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
    #         input_image = rearrange(input_image, "h w c -> 1 c h w").to(next(model.parameters()).device)
    #         cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

    #         uncond = {}
    #         uncond["c_crossattn"] = [null_token]
    #         uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

    #         sigmas = model_wrap.get_sigmas(args.steps)

    #         extra_args = {
    #             "cond": cond,
    #             "uncond": uncond, 
    #             "text_cfg_scale": args.cfg_text,
    #             "image_cfg_scale": args.cfg_image,
    #         }

    #         torch.manual_seed(seed)
    #         z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
    #         z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
    #         x = model.decode_first_stage(z)
    #         x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    #         x = 255.0 * rearrange(x, "1 c h w -> h w c")
    #         # print(x.shape)
    #         edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

    #         edited_image = ImageOps.fit(edited_image, (width, height), method=Image.Resampling.LANCZOS)
    #         edited_image.save(output_dir+'/output_'+input.split('/')[-1].split('.')[0]+'_seed'+str(seed)+'.png')


if __name__ == "__main__":
    main()