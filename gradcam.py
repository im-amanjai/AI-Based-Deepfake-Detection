import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from model import DeepfakeCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_image, class_idx):
        output = self.model(input_image)
        self.model.zero_grad()

        score = output[:, class_idx]
        score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3))
        cam = torch.zeros(self.activations.shape[2:], device=DEVICE)

        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.detach().cpu().numpy()
