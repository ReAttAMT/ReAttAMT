import time
import yaml
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
import copy
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu
from glob import glob
import lpips
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.transforms.functional import to_pil_image


from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.diffusion_utils import get_beta_schedule, denoising_step
from utils.image_processing import *
from utils.model_utils import *
from losses import id_loss
from losses.id_loss import cal_adv_loss
from losses.clip_loss import CLIPLoss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS
from utils.align_utils import run_alignment
import torch.nn.functional as F
from Grad_CAM import compute_gradcam_loss
from utils.text_dic import SRC_TRG_TXT_DIC
from configs.paths_config import DATASET_PATHS, MODEL_PATHS
from remove_makeup import Model_MR


class AttentionAM(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config

        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type

        self.target_id = args.target_img
        self.target_model = args.target_model
        self.ref_id = args.ref_img
        self.cam_save_interval = args.cam_save_interval

        self.target_image, self.test_image, self.target_name = get_target_image(
            self.target_id)
        self.model_list = get_model_list(self.target_model)
        self.src_txt = self.args.src_txt
        self.trg_txt = self.args.trg_txt

        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def clip_finetune(self):
        print(f'   {self.src_txt}')
        print(f'-> {self.trg_txt}')

        # ----------- Model -----------#
        model = DDPM(self.config)
        init_ckpt = torch.load(self.args.model_path)
        learn_sigma = False
        print("Original diffusion Model loaded.")

        model.load_state_dict(init_ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        model_makeup_removal = Model_MR(device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model_mr = model_makeup_removal.init_model()
        print("Makeup Removal Model loaded.")

        # ----------- Optimizer and Scheduler -----------#
        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")
        optim_ft = torch.optim.Adam(
            model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)
        init_opt_ckpt = optim_ft.state_dict()
        scheduler_ft = torch.optim.lr_scheduler.StepLR(
            optim_ft, step_size=1, gamma=self.args.sch_gamma)
        init_sch_ckpt = scheduler_ft.state_dict()

        # ----------- Loss -----------#
        print("Loading losses")
        clip_loss_func = CLIPLoss(
            self.device,
            lambda_direction=1,
            lambda_patch=0,
            lambda_global=0,
            lambda_manifold=0,
            lambda_texture=0,
            clip_model=self.args.clip_model_name)
        loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)
        # ----------- Precompute Latents (image-latent pairs) -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train
        img_lat_pairs_dic = {}
        for mode in ['train', 'test']:
            img_lat_pairs = []
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path, map_location=torch.device('cpu'))
                for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                  f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))
                    tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                  f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config)
                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                            num_workers=self.config.data.num_workers)
                loader = loader_dic[mode]

            for step, (img, _) in enumerate(loader):
                x0 = img.to(self.config.device)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                model.eval()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma)

                            progress_bar.update(1)
                    x_lat = x.clone()
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               learn_sigma=learn_sigma)
                            progress_bar.update(1)

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                if step >= self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)
            print(f'img_lat_pairs saved in {pairs_path}')

        # ----------- Finetune Diffusion Models -----------#
        print("Start finetuning")
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            seq_train = list(range(self.args.t_0))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        model.module.load_state_dict(init_ckpt)
        optim_ft.load_state_dict(init_opt_ckpt)
        scheduler_ft.load_state_dict(init_sch_ckpt)
        clip_loss_func.target_direction = None

        # ----------- Train -----------#
        print("Start training")
        for it_out in range(self.args.n_iter):
            exp_id = os.path.split(self.args.exp)[-1]
            save_name = f'checkpoint/{exp_id}_{self.trg_txt.replace(" ", "_")}-{it_out}.pth'
            if self.args.do_train:
                if os.path.exists(save_name):
                    print(f'{save_name} already exists.')
                    model.module.load_state_dict(torch.load(save_name))
                    continue

                for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic['train']):
                    model.train()
                    time_in_start = time.time()

                    optim_ft.zero_grad()
                    x = x_lat.clone()

                    with tqdm(total=len(seq_train), desc=f"CLIP iteration") as progress_bar:
                        for t_it, (i, j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               eta=self.args.eta,
                                               learn_sigma=learn_sigma)

                            progress_bar.update(1)

                    #clip loss
                    loss_clip = -torch.log((2 - clip_loss_func(x0, self.src_txt, x, self.trg_txt)) / 2)
			#id loss
                    x0 = x0.to(self.device)
                    x = x.to(self.device)
                    loss_l1 = nn.L1Loss()(x0, x)
                    loss_lpips = loss_fn_alex(x0, x)
			#remove makeup
                    x_r = x.clone()
                    x_r = model_makeup_removal.remove(model_mr, x_r)
                    print("removed")

                    tvu.save_image(
                        (x0 + 1) / 2, './sample_real_train/sample_{}.png'.format(step))
                    tvu.save_image(
                        (x + 1) / 2, './sample_fake_train/sample_{}.png'.format(step))
                    tvu.save_image(
                        (x_r + 1)/2, './sample_fake_removed_train/sample_{}.png'.format(step))

	                #adv loss & mr_adv loss
                    loss_adv = 0
                    loss_mr_adv = 0
                    loss_retain_adv = 0
                    targeted_loss_list = []
                    mr_loss_list = []
                    retain_loss_list = []
                    for model_name in list(self.model_list.keys())[:-1]:
                        print(f"********************* model：{model_name} ***********************")
                        untarget_loss_A, target_loss_A, mr_loss_A, retain_loss_A = cal_adv_loss(x, self.target_image, x_r, model_name, self.model_list)
                        targeted_loss_list.append(target_loss_A)
		        untarget_loss_list.append(untarget_loss_A)
                        mr_loss_list.append(mr_loss_A)
                        retain_loss_list.append(retain_loss_A)
                    loss_adv = torch.mean(torch.stack(targeted_loss_list)) 
                    loss_mr_adv = torch.mean(torch.stack(mr_loss_list))
                    loss_retain_adv = torch.mean(torch.stack(retain_loss_list))
		    loss_un_adv = torch.mean(torch.stack(untarget_loss_A))

                    # local loss
                    local_loss_list = []

                    for model_name in list(self.model_list.keys())[:-1]:
                        local_loss_A, attentionmap_x, attentionmap_x_trg = compute_gradcam_loss(x, self.target_image, model_name,
                                                                            self.model_list, self.args.lo_loss_w,
                                                                            self.args.gradcam_loss_v)
                        if step % self.args.cam_save_interval == 0:
                            attention_map_folder = os.path.join(self.args.image_folder, 'attention_maps')
                            os.makedirs(attention_map_folder, exist_ok=True)
                            attentionmap_x_save_path = os.path.join(attention_map_folder,f"x_{model_name}_{step}_{it_out}.png")
                            attentionmap_x_trg_save_path = os.path.join(attention_map_folder,
                                                                    f"trg_x_{model_name}_{it_out}_{step}.png")
                            cv2.imwrite(attentionmap_x_save_path, attentionmap_x)
                            cv2.imwrite(attentionmap_x_trg_save_path, attentionmap_x_trg)

                        local_loss_list.append(local_loss_A)
                    loss_local = torch.mean(torch.stack(local_loss_list))             

                    if it_out < self.args.MT_iter_without_adv:
                        loss = (self.args.clip_loss_w * loss_clip +
                                self.args.lpips_loss_w * loss_lpips +
                                self.args.l1_loss_w * loss_l1)
                    else:
                        loss = (self.args.clip_loss_w * loss_clip +
                                self.args.lpips_loss_w * loss_lpips +
                                self.args.l1_loss_w * loss_l1 +
                                self.args.adv_loss_w * loss_adv 
				+ self.args.un_adv_loss_w * loss_un_adv
                                + self.args.local_loss_w * loss_local
				                + self.args.loss_mr_adv_w * loss_mr_adv
                                + self.args.loss_retain_adv_w * loss_retain_adv
                                )

                    loss.backward()
                    optim_ft.step()

                    print(f"CLIP {step}-{it_out}@LOSS: clip: {loss_clip:.3f}, lpips: {loss_lpips.item():.3f}, l1: {loss_l1.item():.3f}, adv: {loss_adv:.3f}, local: {loss_local.item():.3f}, mr_adv: {loss_mr_adv.item():.3f}, retain_adv: {loss_retain_adv.item():.3f}" )

                    if self.args.save_train_image:
                        tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'train_{step}_2_clip_{self.trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_train_step}.png'))
                    time_in_end = time.time()
                    print(f"Training for 1 image takes {time_in_end - time_in_start:.4f}s")

                    if step == self.args.n_train_img - 1:
                        break

                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), save_name)
                else:
                    torch.save(model.state_dict(), save_name)
                print(f'Model {save_name} is saved.')
                scheduler_ft.step()

            # ----------- Eval -----------#
            if self.args.do_test:
                if not self.args.do_train:
                    print(save_name)
                    #model.module.load_state_dict(torch.load(save_name))
                    model.module.load_state_dict(torch.load(f'checkpoint/{exp_id}_{self.trg_txt.replace(" ", "_")}-5.pth'))
                if it_out == 5:
                    model.eval()
                    img_lat_pairs = img_lat_pairs_dic[mode]
                    FAR01 = 0
                    FAR001 = 0
                    FAR0001 = 0
                    mr_FAR01 = 0
                    mr_FAR001 = 0
                    mr_FAR0001 = 0
                    total = 0
                    sum_psnr = 0
                    sum_ssim = 0
                    if self.target_id == 0:
                        min_step = 0
                        max_step = 250
                    elif self.target_id == 1:
                        min_step = 250
                        max_step = 500
                    elif self.target_id == 2:
                        min_step = 500
                        max_step = 750
                    else:
                        min_step = 750
                        max_step = 1030

                    for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs):
                        if min_step <= step < max_step:
                            x0 = x0.to(self.device)
                            x_id = x_id.to(self.device)
                            x_lat = x_lat.to(self.device)
                            with torch.no_grad():
                                x = x_lat
                                with tqdm(total=len(seq_test), desc=f"Eval iteration") as progress_bar:
                                    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                                        t = (torch.ones(n) * i).to(self.device)
                                        t_next = (torch.ones(n) * j).to(self.device)

                                        x = denoising_step(x, t=t, t_next=t_next, models=model,
                                                        logvars=self.logvar,
                                                        sampling_type=self.args.sample_type,
                                                        b=self.betas,
                                                        eta=self.args.eta,
                                                        learn_sigma=learn_sigma)

                                        progress_bar.update(1)

                            th_dict = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
                                    'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878)}
                            # remove makeup
                            x_r = x.clone()
                            x_r = model_makeup_removal.remove(model_mr, x_r)
                            print("removed")
                            x0 = (x0 + 1) / 2
                            x = (x + 1) / 2
                            x_r = (x_r + 1) / 2
                            tvu.save_image(
                                x_r, './sample_fake_removed_test/sample_{}.png'.format(step))
                            tvu.save_image(
                                x0, './sample_real_test/sample_{}.png'.format(step))
                            tvu.save_image(
                                x, './sample_fake_test/sample_{}.png'.format(step))
                            x0_ps = to_pil_image(x0.squeeze(0).cpu())
                            x_ps = to_pil_image(x.squeeze(0).cpu())
                            psnr = compare_psnr(np.array(x0_ps), np.array(x_ps), data_range=255)
                            ssim = compare_ssim(np.array(x0_ps), np.array(x_ps), data_range=255, channel_axis=-1)
                            sum_psnr += psnr
                            sum_ssim += ssim

                            for test_model in list(self.model_list.keys())[-1:]:
                                print(f"********************* model：{test_model} ***********************")
                                size = self.model_list[test_model][0]
                                test_model_ = self.model_list[test_model][1]
                                target_embbeding = test_model_(
                                    (F.interpolate(self.test_image, size=size, mode='bilinear')))

                                ae_embbeding = test_model_(
                                    (F.interpolate(x, size=size, mode='bilinear')))
                                mr_embbeding = test_model_(
                                    (F.interpolate(x_r, size=size, mode='bilinear')))

                                cos_simi = torch.cosine_similarity(
                                    ae_embbeding, target_embbeding)
                                cos_simi_mr = torch.cosine_similarity(
                                    mr_embbeding, target_embbeding)

                                #makeup_removal
                                if cos_simi_mr.item() > th_dict[test_model][0]:
                                    mr_FAR01 += 1
                                if cos_simi_mr.item() > th_dict[test_model][1]:
                                    mr_FAR001 += 1
                                if cos_simi_mr.item() > th_dict[test_model][2]:
                                    mr_FAR0001 += 1
                                #makeup
                                if cos_simi.item() > th_dict[test_model][0]:
                                    FAR01 += 1
                                if cos_simi.item() > th_dict[test_model][1]:
                                    FAR001 += 1
                                if cos_simi.item() > th_dict[test_model][2]:
                                    FAR0001 += 1

                                total += 1
                            print(f"Eval {step}-{it_out}")

                            #if step == self.args.n_test_img - 1:
                                #break
                        else:
                            continue

                    print(f"total{total}")
                    print("Before mr: ASR in FAR@0.1: {:.4f}, ASR in FAR@0.01: {:.4f}, ASR in FAR@0.001: {:.4f}".
                          format(FAR01 / total, FAR001 / total, FAR0001 / total))
                    print("After mr:  ASR in FAR@0.1: {:.4f}, ASR in FAR@0.01: {:.4f}, ASR in FAR@0.001: {:.4f}".
                          format(mr_FAR01 / total, mr_FAR001 / total, mr_FAR0001 / total))
                    print("PSNR: {:.4f}, SSIM: {:.4f}".
                          format(sum_psnr / total, sum_ssim / total))
                    if self.args.do_test == 1 and self.args.do_train == 0:
                        break
                    break

    def edit_one_image(self):
        # ----------- Data -----------#
        n = self.args.bs_test
        try:
            img = run_alignment(self.args.img_path,
                                output_size=self.config.data.image_size)
        except:
            img = Image.open(self.args.img_path).convert("RGB")

        img = img.resize((self.config.data.image_size,
                          self.config.data.image_size), Image.ANTIALIAS)
        img = np.array(img) / 255
        img = torch.from_numpy(img).type(torch.FloatTensor).permute(
            2, 0, 1).unsqueeze(dim=0).repeat(n, 1, 1, 1)
        img = img.to(self.config.device)
        tvu.save_image(img, os.path.join(
            self.args.image_folder, f'0_orig.png'))
        x0 = (img - 0.5) * 2.

        models = []

        model_paths = [None, self.args.model_path]

        for model_path in model_paths:
            model_i = DDPM(self.config)
            if model_path:
                ckpt = torch.load(model_path)
            else:
                ckpt = torch.load('pretrained/celeba_hq.ckpt')
            learn_sigma = False
            model_i.load_state_dict(ckpt)
            model_i.to(self.device)
            model_i = torch.nn.DataParallel(model_i)
            model_i.eval()
            print(f"{model_path} is loaded.")
            models.append(model_i)

        with torch.no_grad():
            # ---------------- Invert Image to Latent in case of Deterministic Inversion process -------------------#
            if self.args.deterministic_inv:
                x_lat_path = os.path.join(
                    self.args.image_folder, f'x_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth')
                if not os.path.exists(x_lat_path):
                    seq_inv = np.linspace(
                        0, 1, self.args.n_inv_step) * self.args.t_0
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
                        torch.save(x_lat, x_lat_path)
                else:
                    print('Latent exists.')
                    x_lat = torch.load(x_lat_path)

            # ----------- Generative Process -----------#
            print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
                  f" Steps: {self.args.n_test_step}/{self.args.t_0}")
            if self.args.n_test_step != 0:
                seq_test = np.linspace(
                    0, 1, self.args.n_test_step) * self.args.t_0
                seq_test = [int(s) for s in list(seq_test)]
                print('Uniform skip type')
            else:
                seq_test = list(range(self.args.t_0))
                print('No skip')
            seq_test_next = [-1] + list(seq_test[:-1])

            for it in range(self.args.n_iter):
                if self.args.deterministic_inv:
                    x = x_lat.clone()
                else:
                    e = torch.randn_like(x0)
                    a = (1 - self.betas).cumprod(dim=0)
                    x = x0 * a[self.args.t_0 - 1].sqrt() + e * \
                        (1.0 - a[self.args.t_0 - 1]).sqrt()
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'1_lat_ninv{self.args.n_inv_step}.png'))

                with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)

                        x = denoising_step(x, t=t, t_next=t_next, models=models,
                                           logvars=self.logvar,
                                           sampling_type=self.args.sample_type,
                                           b=self.betas,
                                           eta=self.args.eta,
                                           learn_sigma=learn_sigma,
                                           ratio=self.args.model_ratio)

                        # added intermediate step vis
                        if (i - 99) % 100 == 0:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'2_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_{i}_it{it}.png'))
                        progress_bar.update(1)

                x0 = x.clone()
                if self.args.model_path:
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                               f"3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}_{self.args.model_path.split('/')[-1].replace('.pth', '')}.png"))
                else:
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                               f'3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}.png'))
