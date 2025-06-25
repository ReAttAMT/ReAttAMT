import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from assets.models import irse, ir152, facenet
from losses.id_loss import cal_adv_loss
import insightface
from insightface.app import FaceAnalysis

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor):

        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if output is None or output.numel() == 0:
            raise ValueError("Model output is empty or None")

        target_class = output.argmax(dim=1)

        one_hot = torch.zeros_like(output).scatter_(1, target_class.view(-1, 1), 1)
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured")

        alpha = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (alpha * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max == cam_min:
            return np.zeros_like(cam)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam


def overlay_heatmap(img, cam, alpha=0.6):
    if isinstance(img, torch.Tensor):
        img = img.squeeze(0).detach().cpu().numpy()  
        img = np.transpose(img, (1, 2, 0))  
        img = (img * 255).astype(np.uint8)  

    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("overlay_heatmap received an invalid image!")
    
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)

def load_face(img):
    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError("No Face!")
    return img, faces[0]

def init_face_analysis():
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app
app = init_face_analysis()  
MODEL_CACHE = {}
def get_fr_model(model, device):
    if model in MODEL_CACHE:
        return MODEL_CACHE[model]
    
    if model == 'ir152':
        fr_model = ir152.IR_152((112, 112))
        fr_model.load_state_dict(torch.load('./assets/models/ir152.pth', map_location=device))
        target_layer = fr_model.body[-1].res_layer[3]
    elif model == 'irse50':
        fr_model = irse.Backbone(50, 0.6, 'ir_se')
        fr_model.load_state_dict(torch.load('./assets/models/irse50.pth', map_location=device))
        target_layer = fr_model.body[-3].res_layer[3]
    elif model == 'facenet':
        fr_model = facenet.InceptionResnetV1(num_classes=8631, device=device)
        fr_model.load_state_dict(torch.load('./assets/models/facenet.pth', map_location=device))
        target_layer = fr_model.block8.conv2d
    elif model == 'mobile_face':
        fr_model = irse.MobileFaceNet(512)
        fr_model.load_state_dict(torch.load('./assets/models/mobile_face.pth', map_location=device))
        target_layer = fr_model.conv_5.model[-1].conv_dw.conv
    else:
        raise ValueError(f"Unknown model: {model}")
    
    fr_model.to(device)
    fr_model.eval()
    MODEL_CACHE[model] = (fr_model, target_layer)
    return fr_model, target_layer
####################################################################################################
def compute_gradcam(model,img):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    fr_model, target_layer = get_fr_model(model, device)
    grad_cam = GradCAM(fr_model, target_layer)
    cam = grad_cam.generate(img)
    torch.cuda.empty_cache()
    cam_tensor = torch.tensor(cam, dtype=torch.float32, device=device)
    MASK = (cam_tensor > 0.5).float()
    return cam, cam_tensor, MASK


def compute_gradcam_loss(makeup_img, target_img, model, target_models, lo_loss_w, gradcam_loss_v):

    input_size = target_models[model][0]
    fr_model = target_models[model][1]
    makeup_resize = F.interpolate(makeup_img, size=input_size, mode='bilinear')
    target_resize = F.interpolate(target_img, size=input_size, mode='bilinear')
    emb_makeup = fr_model(makeup_resize)
    emb_target = fr_model(target_resize).detach()

    cam_x0, cam_tensor_x0, MASK_target = compute_gradcam(model, target_resize)
    cam_x, cam_tensor_x, MASK_makeup = compute_gradcam(model, makeup_resize)

    MASK_target = F.interpolate(MASK_target.unsqueeze(0).unsqueeze(0), size=(1,512), mode='bilinear').squeeze()
    MASK_makeup = F.interpolate(MASK_makeup.unsqueeze(0).unsqueeze(0), size=(1,512), mode='bilinear').squeeze()
    MASK_target = MASK_target.view(1, 512)
    MASK_makeup = MASK_makeup.view(1, 512)
    MASK_target = F.normalize(MASK_target, p=2, dim=-1)
    MASK_makeup = F.normalize(MASK_makeup, p=2, dim=-1)

    gradcam_loss = nn.MSELoss()(cam_tensor_x0, cam_tensor_x)
    lo_loss = 1 - torch.mean(torch.sum(torch.mul(MASK_target * emb_target, MASK_makeup * emb_makeup), dim=1) / ((MASK_target * emb_target).norm(dim=1) + 1e-8) / ((MASK_makeup * emb_makeup).norm(dim=1) + 1e-8))
    local_loss = gradcam_loss + lo_loss_w * (torch.sigmoid(gradcam_loss_v - gradcam_loss)) * lo_loss
    print(f"sigmoid(gradcam_loss_v - gradcam_loss): {torch.sigmoid(gradcam_loss_v - gradcam_loss)}")
    #local_loss = lo_loss
    print(f"1. GradCam_Loss: {gradcam_loss.item():.3f} 2. Lo_loss: {lo_loss.item():.3f} ")

    makeup_resize_np = makeup_resize.squeeze(0).detach().cpu().numpy()
    makeup_resize_np = np.transpose(makeup_resize_np, (1, 2, 0)) 
    makeup_resize_np = (makeup_resize_np * 255).astype(np.uint8) 

    att_map = overlay_heatmap(makeup_resize, cam_x, alpha=0.7)
    att_map_x0 = overlay_heatmap(target_resize, cam_x0, alpha=0.7)
    return local_loss, att_map, att_map_x0

