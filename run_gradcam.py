import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from model import DeepfakeCNN
from gradcam import GradCAM

IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = DeepfakeCNN().to(DEVICE)
model.load_state_dict(torch.load("models/deepfake_cnn.pth", map_location=DEVICE))
model.eval()

# Grad-CAM on last conv layer
gradcam = GradCAM(model, model.features[3])

# Image transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load image
img_path = "raw_data/real/images.jpg"   # 🔁 change image name if needed
original = Image.open(img_path).convert("RGB")
input_tensor = transform(original).unsqueeze(0).to(DEVICE)

# Prediction
output = model(input_tensor)
class_idx = torch.argmax(output, dim=1).item()

# Generate Grad-CAM
cam = gradcam.generate(input_tensor, class_idx)
cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

# Overlay heatmap
img_np = np.array(original.resize((IMAGE_SIZE, IMAGE_SIZE)))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

# Save result
cv2.imwrite("gradcam_output.jpg", overlay)
print("✅ Grad-CAM saved as gradcam_output.jpg")
