import time
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu
import lpips

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.diffusion_utils import get_beta_schedule, denoising_step
from losses import id_loss
from losses.clip_loss import CLIPLoss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS
from utils.align_utils import run_alignment


class Model_MR(object):
    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = 'fixedsmall'
        betas = get_beta_schedule(
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def init_model(self):
        models = []
        model_paths = [None, "checkpoint/test_MR_MT_t300_ninv40_ngen6_id1_l12_lr8e-06_face_without_makeup-6.pth"]
        for model_path in model_paths:
            model_i = i_DDPM()
            if model_path:
                ckpt = torch.load(model_path)
            else:
                ckpt = torch.load("pretrained/makeup.pt")
            learn_sigma = True
            model_i.load_state_dict(ckpt)
            model_i.to(self.device)
            model_i = torch.nn.DataParallel(model_i)
            model_i.eval()
            print(f"{model_path} is loaded.")
            models.append(model_i)
        return models

    def remove(self, models, img):
        # ----------- Data -----------#
        n = 1
		#img = img.convert("RGB")
        #img = img.resize((self.config.data.image_size, self.config.data.image_size), Image.ANTIALIAS)
        #img = np.array(img)/255
        #img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(dim=0).repeat(n, 1, 1, 1)
        #img = img.to(self.config.device)
        #x0 = (img - 0.5) * 2.
        x0 = img
        learn_sigma = True
        with torch.no_grad():
            # ---------------- Invert Image to Latent in case of Deterministic Inversion process -------------------#
            seq_inv = np.linspace(0, 1, 40) * (300)
            seq_inv = [int(s) for s in list(seq_inv)]
            seq_inv_next = [-1] + list(seq_inv[:-1])
            x = x0.clone()
            with tqdm(total=len(seq_inv), desc=f"Inversion process ") as progress_bar:
                for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                    t = (torch.ones(n) * i).to(self.device)
                    t_prev = (torch.ones(n) * j).to(self.device)

                    x = denoising_step(x, t=t, t_next=t_prev, models=models[0],
                                        logvars=self.logvar,
                                        sampling_type='ddim',
                                        b=self.betas,
                                        eta=0,
                                        learn_sigma=learn_sigma,
                                        ratio=0,
                                        )

                    progress_bar.update(1)
                x_lat = x.clone()

            # ----------- Generative Process -----------#
            print(f"Sampling type: {'ddim'.upper()} with eta 0.0, "
                  f" Steps: 40/300")
            
            seq_test = np.linspace(0, 1, 40) * 300
            seq_test = [int(s) for s in list(seq_test)]
            print('Uniform skip type')
        
            seq_test_next = [-1] + list(seq_test[:-1])
                
            x = x_lat.clone()

            with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                    t = (torch.ones(n) * i).to(self.device)
                    t_next = (torch.ones(n) * j).to(self.device)

                    x = denoising_step(x, t=t, t_next=t_next, models=models,
                                        logvars=self.logvar,
                                        sampling_type='ddim',
                                        b=self.betas,
                                        eta=0.0,
                                        learn_sigma=learn_sigma,
                                        ratio=1)
                    progress_bar.update(1)

            x0 = x.clone()
        return x0