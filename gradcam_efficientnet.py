import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register()

    def _register(self):
        def fwd_hook(_, __, output):
            self.activations = output

        def bwd_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(fwd_hook)
        # recommended API
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, x, class_idx):
        out = self.model(x)
        self.model.zero_grad()
        out[:, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3))
        cam = (weights[:, :, None, None] * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam
