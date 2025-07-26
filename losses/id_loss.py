import torch
from torch import nn
from configs.paths_config import MODEL_PATHS
from models.insight_face.model_irse import Backbone, MobileFaceNet
import torch.nn.functional as F



def cos_simi(emb_1, emb_2):
    return torch.mean(torch.sum(torch.mul(emb_2, emb_1), dim=1) / emb_2.norm(dim=1) / emb_1.norm(dim=1))

def cal_adv_loss(untarget, source, target, x_r, model_name, target_models):
    input_size = target_models[model_name][0]
    fr_model = target_models[model_name][1]

    source_resize = F.interpolate(source, size=input_size, mode='bilinear')
    target_resize = F.interpolate(target, size=input_size, mode='bilinear')
    untarget_resize = F.interpolate(untarget, size=input_size, mode='bilinear')
    x_r_resize = F.interpolate(x_r, size=input_size, mode='bilinear')

    emb_source = fr_model(source_resize)
    emb_target = fr_model(target_resize).detach()
    emb_untarget = fr_model(untarget_resize).detach()
    emb_x_r = fr_model(x_r_resize)

    id_retain = emb_x_r - emb_source
    mask_retain = (id_retain.abs() < 0.1).float()

    cos_loss = 1 - cos_simi(emb_source, emb_target)
    untarget_loss = 1 - cos_simi(emb_source, emb_untarget)
    mr_cos_loss = 1 - cos_simi(emb_x_r, emb_target)
    retain_loss = 1 - cos_simi(emb_x_r * mask_retain, emb_target * mask_retain)
    

    return untarget_loss, cos_loss, mr_cos_loss, retain_loss
