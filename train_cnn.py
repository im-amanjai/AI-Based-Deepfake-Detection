import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model import DeepfakeCNN   # ✅ import model properly

# -------------------------
# Config
# -------------------------
BATCH_SIZE = 4
EPOCHS = 5
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Main Training Function
# -------------------------
def train():
    # -------------------------
    # Transforms
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # -------------------------
    # Dataset
    # -------------------------
    train_data = datasets.ImageFolder("dataset/train", transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # -------------------------
    # Model
    # -------------------------
    model = DeepfakeCNN().to(DEVICE)

    # -------------------------
    # Loss & Optimizer
    # -------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Accuracy={accuracy:.2f}%")

    # -------------------------
    # Save Model
    # -------------------------
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/deepfake_cnn.pth")
    print("✅ Model training completed and saved.")

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    train()
