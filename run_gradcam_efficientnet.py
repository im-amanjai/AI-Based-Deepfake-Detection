import torch, cv2, numpy as np
from PIL import Image
from torchvision import transforms
import timm
from gradcam_efficientnet import GradCAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

# Load EfficientNet-B0
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/efficientnet_deepfake.pth", map_location=DEVICE))
model.eval().to(DEVICE)

# Target last conv block
target_layer = model.conv_head
gradcam = GradCAM(model, target_layer)

# Transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 🔁 Change image path as needed
img_path = "raw_data/real/00000.jpg"
orig = Image.open(img_path).convert("RGB")
x = transform(orig).unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    logits = model(x)
class_idx = logits.argmax(dim=1).item()

# Grad-CAM
cam = gradcam.generate(x, class_idx)[0].detach().cpu().numpy()
cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

# Overlay
img_np = np.array(orig.resize((IMAGE_SIZE, IMAGE_SIZE)))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

cv2.imwrite("gradcam_efficientnet.jpg", overlay)
print("✅ Saved: gradcam_efficientnet.jpg")
