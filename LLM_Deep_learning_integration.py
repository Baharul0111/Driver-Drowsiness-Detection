import os
import sys
import base64
import uuid
import random
import requests
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVDINO_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nv-dinov2"
headers_auth = f"Bearer {NVIDIA_API_KEY}"
random.seed(42)

def _upload_asset(input_data, description):
    assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
    headers = {
        "Authorization": headers_auth,
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    s3_headers = {
        "x-amz-meta-nvcf-asset-description": description,
        "content-type": "image/jpeg",
    }
    payload = {"contentType": "image/jpeg", "description": description}

    response = requests.post(assets_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    asset_url = response.json()["uploadUrl"]
    asset_id = response.json()["assetId"]

    response = requests.put(
        asset_url,
        data=input_data,
        headers=s3_headers,
        timeout=300,
    )
    response.raise_for_status()
    return uuid.UUID(asset_id)

def get_image_embeddings(image_path):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    if len(image_b64) < 200_000:
        payload = {
            "messages": [
                {
                    "content": {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                }
            ]
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": headers_auth,
            "Accept": "application/json",
        }
    else:
        with open(image_path, "rb") as f:
            asset_id = _upload_asset(f.read(), "Input Image")
        payload = {"messages": []}
        headers = {
            "Content-Type": "application/json",
            "NVCF-INPUT-ASSET-REFERENCES": str(asset_id),
            "Authorization": headers_auth,
        }

    response = requests.post(NVDINO_URL, headers=headers, json=payload)
    response.raise_for_status()

    embedding = response.json()["messages"][0]["content"]["embedding"]
    return embedding

def prepare_dataset(data_path):
    categories = ["Closed", "no_yawn", "Open", "yawn"]
    train_images, val_images, test_images = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for idx, category in enumerate(categories):
        category_folder = os.path.join(data_path, category)
        images = os.listdir(category_folder)
        random.shuffle(images)

        train_images += [os.path.join(category_folder, img) for img in images[:50]]
        train_labels += [idx] * 50

        val_images += [os.path.join(category_folder, img) for img in images[50:70]]
        val_labels += [idx] * 20

        test_images += [os.path.join(category_folder, img) for img in images[70:90]]
        test_labels += [idx] * 20

    return train_images, val_images, test_images, train_labels, val_labels, test_labels

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, criterion, optimizer, dataloader, device):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device).float(), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def main(data_path):
    train_images, val_images, test_images, train_labels, val_labels, test_labels = prepare_dataset(data_path)

    print("Extracting train embeddings...")
    train_embeddings = [get_image_embeddings(img) for img in tqdm(train_images)]

    print("Extracting validation embeddings...")
    val_embeddings = [get_image_embeddings(img) for img in tqdm(val_images)]

    print("Extracting test embeddings...")
    test_embeddings = [get_image_embeddings(img) for img in tqdm(test_images)]

    train_dataset = TensorDataset(torch.tensor(train_embeddings), torch.tensor(train_labels))
    val_dataset = TensorDataset(torch.tensor(val_embeddings), torch.tensor(val_labels))
    test_dataset = TensorDataset(torch.tensor(test_embeddings), torch.tensor(test_labels))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    input_dim = len(train_embeddings[0])
    hidden_dim, output_dim = 128, 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        train_loss = train_model(model, criterion, optimizer, train_loader, device)
        val_accuracy = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%")

    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    torch.save(model.state_dict(), "drowsiness_detection_model.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    data_path = "data"
    main(data_path)
