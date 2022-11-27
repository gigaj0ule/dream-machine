import argparse, os, re, glob, json
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
#from torch import autocast
from torch.cuda.amp import autocast
from contextlib import contextmanager, nullcontext

import sys
sys.path.append('.')
from ldm.util import instantiate_from_config

from optimUtils import split_weighted_subprompts, logger
from transformers import logging
# from samplers import CompVisDenoiser
logging.set_verbosity_error()


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


# Entry

if __name__ == "__main__":

    config = "dream_machine/v1-inference.yaml"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", 
        type=str, 
        nargs="?", 
        default=None,
        help="Prompt to dream"
    )

    parser.add_argument(
        "--outdir", 
        type=str, 
        nargs="?", 
        help="Dir to write results to", 
        default="output_txt2img"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps",
    )

    parser.add_argument(
        "--fixed_code",
        action="store_true",
        help="If enabled, uses the same starting code across samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="DDIM eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="Sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="Image height, must be multiple of 64",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=768,
        help="Image width, must be multiple of 64",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="Latent channels",
    )

    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="Downsampling factor",
    )
    
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="How many samples to produce for each given prompt. A.k.a. batch size",
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="Rows in the grid (default: n_samples)",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    
    parser.add_argument(
        "--from-file",
        type=str,
        help="If specified, load prompts from this file",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="AI seed (for reproducible sampling)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Specify GPU (cuda/cuda:0/cuda:1/...)",
    )

    parser.add_argument(
        "--unet_bs",
        type=int,
        default=1,
        help="Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )",
    )
    
    parser.add_argument(
        "--turbo",
        action="store_true",
        default=True,
        help="Reduces inference time on the expense of 1GB VRAM",
    )

    parser.add_argument(
        "--precision", 
        type=str,
        help="Evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--format",
        type=str,
        help="Output image format",
        choices=["jpg", "png"],
        default="png",
    )

    parser.add_argument(
        "--sampler",
        type=str,
        help="Sampler type (default plms)",
        choices=["ddim", "plms","heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms"],
        default="plms",
    )

    # Get the AI Model
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to checkpoint of model",
        default=None
    )
    
    # Parse args
    varargs = parser.parse_args()

    # Check for saved settings
    try:

        with open("txt2img.json", "r") as f:
            saved_args = json.load(f)

        print(saved_args)

    except:
        saved_args = None


    # If no AI model exists...
    if varargs.ckpt == None:

        # Model directory
        model_directory = "./ai_models"

        # Scan model directory for models
        print(f"╔══════════════════")
        print(f"║ No ckpt model was specified with --ckpt")
        print(f"║")
        print(f"║ Here are the available models in {model_directory}:")
        print(f"║")

        file_selector = 0

        list_of_ckpts = glob.glob(f"{model_directory}/*.ckpt")

        for file in list_of_ckpts:
            
            print(f"║ {file_selector}: {file}")
            file_selector += 1

        # Prompt for selection
        print(f"║")
        file_selector_user_prompt = input(f"║ Please make your selection... ")

        if file_selector_user_prompt == '':
            if(saved_args is not None):
                
                varargs.ckpt = saved_args.get('ckpt')

        else:
            file_selector_user_prompt = int(file_selector_user_prompt)
            
            varargs.ckpt = str(list_of_ckpts[file_selector_user_prompt])

        print(f"║")
        print(f"║ You have selected {varargs.ckpt}")
        print(f"╚══════════════════")


    # If no prompt exists...
    if varargs.prompt == None:

        print(f"╔══════════════════")
        print(f"║ No prompt was specified with --prompt")
        print(f"║")

        ai_prompt_user_prompt = str(input(f"║ What do you want to dream? "))

        varargs.prompt = str(ai_prompt_user_prompt)

        # If empty, reload last
        if(len(varargs.prompt) < 1 and saved_args is not None):
            varargs.prompt = saved_args.get('prompt')                
            
        print(f"║")
        print(f"║ Let's dream about '{varargs.prompt}'")
        print(f"╚══════════════════")

    tic = time.time()
    os.makedirs(varargs.outdir, exist_ok=True)
    outpath = varargs.outdir
    grid_count = len(os.listdir(outpath)) - 1

    # If no seed exists...
    if varargs.seed == None:

        print(f"╔══════════════════")
        print(f"║ No seed was specified with --seed")
        print(f"║")
        varargs.seed = int(input(f"║ Enter seed, or press enter for random: ") or randint(0, 1000))
        print(f"║")
        print(f"║ Seeding AI with '{varargs.seed}'")
        print(f"╚══════════════════")


    # Data to be written
    json_dictionary = {
        "prompt": varargs.prompt,
        "ckpt": varargs.ckpt,
        "seed": varargs.seed
    }
    
    # Serializing json
    json_object = json.dumps(json_dictionary, indent=4)
    
    # Writing to .json
    with open("txt2img.json", "w") as outfile:
        outfile.write(json_object)

    # Seed AI
    #with NoStdStreams(): 
    seed_everything(varargs.seed)

    # Logging
    logger(vars(varargs), log_csv = f"{varargs.outdir}/txt2img_log.csv")

    sd = load_model_from_config(f"{varargs.ckpt}")

    li, lo = [], []
    for key, value in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    model.unet_bs = varargs.unet_bs
    model.cdevice = varargs.device
    model.turbo = varargs.turbo

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = varargs.device

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    if varargs.device != "cpu" and varargs.precision == "autocast":
        model.half()
        modelCS.half()

    start_code = None
    if varargs.fixed_code:
        start_code = torch.randn([varargs.n_samples, varargs.C, varargs.H // varargs.f, varargs.W // varargs.f], device=varargs.device)


    batch_size = varargs.n_samples
    
    n_rows = varargs.n_rows if varargs.n_rows > 0 else batch_size

    if not varargs.from_file:
        assert varargs.prompt is not None
        prompt = varargs.prompt
        print(f"╔══════════════════")
        print(f"║ Using prompt: {prompt}")
        print(f"╚══════════════════")
        data = [batch_size * [prompt]]

    else:
        print(f"╔══════════════════")
        print(f"║ Reading prompts from {varargs.from_file}")
        with open(varargs.from_file, "r") as f:
            text = f.read()
            print(f"║ Using prompt: {text.strip()}")
            data = text.splitlines()
            data = batch_size * list(data)
            data = list(chunk(sorted(data), batch_size))
        print(f"╚══════════════════")


    if varargs.precision == "autocast" and varargs.device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    seeds = ""
    with torch.no_grad():

        all_samples = list()
        
        print(f"╔══════════════════")
        for n in trange(varargs.n_iter, desc="║ Sampling"):
            for prompts in tqdm(data, desc="║ data"):

                sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompts[0])))[:150]
                os.makedirs(sample_path, exist_ok=True)
                base_count = len(os.listdir(sample_path))

                #with precision_scope("cuda"):
                with precision_scope(True):
                    modelCS.to(varargs.device)
                    uc = None
                    if varargs.scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    shape = [varargs.n_samples, varargs.C, varargs.H // varargs.f, varargs.W // varargs.f]

                    if varargs.device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    samples_ddim = model.sample(
                        S=varargs.ddim_steps,
                        conditioning=c,
                        seed=varargs.seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=varargs.scale,
                        unconditional_conditioning=uc,
                        eta=varargs.ddim_eta,
                        x_T=start_code,
                        sampler = varargs.sampler,
                    )

                    modelFS.to(varargs.device)

                    print(samples_ddim.shape)
                    print(f"║ saving images")
                    for i in range(batch_size):

                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")

                        this_image_path = os.path.join(sample_path, "seed_" + str(varargs.seed) + "_" + f"{base_count:05}")

                        # Save this image
                        Image.fromarray(x_sample.astype(np.uint8)).save( f"{this_image_path}.{varargs.format}" )

                        # Save image metadata
                        json_object = json.dumps(json_dictionary, indent=4)
                        
                        # Writing to .json
                        with open(f"{this_image_path}.json", "w") as outfile:
                            outfile.write(json_object)

                        # save most recent image to local file
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(outpath, f"latest.{varargs.format}")
                        )

                        seeds += str(varargs.seed) + ","
                        varargs.seed += 1
                        base_count += 1


                    if varargs.device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    del samples_ddim
                    print(f"║ memory_final = ", torch.cuda.memory_allocated() / 1e6)

        print(f"╚══════════════════")

    toc = time.time()

    time_taken = (toc - tic) / 60.0

    print(f"╔══════════════════")
    print(("║ Samples finished in {0:.2f} minutes and exported to ").format(time_taken) + sample_path)
    print(f"║ Seeds used = {seeds[:-1]}")
    print(f"╚══════════════════")
