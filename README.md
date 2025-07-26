#  Facial Privacy Protection: Reliable Attention-guided Adversarial Makeup Transfer via Diffusion Models(ReAttAMT)
Official PyTorch implementation of paper "Facial Privacy Protection: Reliable Attention-guided Adversarial Makeup Transfer via Diffusion Models".
## Setup
### Environment
```
python=3.8
```

```
pip install -r requirement.txt
pip install git+https://github.com/openai/CLIP.git
```
## Models & datasets
Pretrained diffusion model: Download `celeba-hq.ckpt` [here](https://drive.google.com/drive/folders/1L8caY-FVzp9razKMuAt37jCcgYh3fjVU) and unzip into the `ReAttAMT/pretrained` folder.

FR models, target image: Download [here](https://drive.google.com/file/d/1IKiWLv99eUbv3llpj-dOegF3O7FWW29J/view) and unzip into the `ReAttAMT/assets` folder.

CelebA-HQ: Download [here](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view) and unzip into the `ReAttAMT/assets/datasets` folder.
## Train & Test
Generate adversarial makeup images:
```
python main.py --makeup_transfer --config celeba.yml --exp ./runs/test --n_train_img 200 --n_test_img 1000 --n_iter 6 --t_0 100 --n_inv_step 40 --n_train_step 6 --n_test_step 6 --lr_clip_finetune 4e-6 --MT_iter_without_adv 2 --model_path pretrained/celeba_hq.ckpt --do_train 1 --do_test 1 --target_img 1 --target_model 0
```
`target_img`: Choose one of the four target identities to attack.

`target_model`: Choose the target FR models to attack, including `IRSE50, IR152, Mobileface, Facenet`.

The makeup text prompt can be customized in `main.py` or by `--src_txt 'xxx'` `--trg_txt 'xxx'`.

Untargeted(Dodging) attack can be implemented by `--un_adv_loss_w = 0.4`.

Output images in `ReAttAMT/sample_real_train`, `ReAttAMT/sample_fake_train`, `ReAttAMT/sample_real_test`, `ReAttAMT/sample_fake_test`, `ReAttAMT/sample_real_train`, `ReAttAMT/sample_fake_train`, `ReAttAMT/sample_real_test`, `ReAttAMT/sample_fake_test`.

Attention maps in `ReAttAMT/runs`.

## Acknowledge
Some of the codes are built upon [DiffAM](https://github.com/HansSunY/DiffAM) and [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP)
## Citation
```
TBD
```
