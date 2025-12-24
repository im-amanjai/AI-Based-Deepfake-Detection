import torch
from torchvision import transforms
from PIL import Image
import sys

# -------------------------
# Config
# -------------------------
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/deepfake_cnn.pth"

# -------------------------
# Load Model
# -------------------------
from train_cnn import DeepfakeCNN
model = DeepfakeCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -------------------------
# Predict Function
# -------------------------
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return "FAKE" if predicted.item() == 1 else "REAL"

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    img_path = sys.argv[1]
    result = predict(img_path)
    print("Prediction:", result)
