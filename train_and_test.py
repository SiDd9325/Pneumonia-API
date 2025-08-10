import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models import DenseNet121_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report

# 1. Config
DATA_DIR = "chest_xray"
BATCH_SIZE = 16
NUM_EPOCHS = 1  # 
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

#  3. Datasets (load first, then subset)
train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_data = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
test_data = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

#  4. Subset for fast execution (test mode)
train_data = Subset(train_data, range(500))
val_data = Subset(val_data, range(100))
test_data = Subset(test_data, range(100))

#  5. DataLoaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=0)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=0)

#  6. Model
weights = DenseNet121_Weights.DEFAULT
model = models.densenet121(weights=weights)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model = model.to(DEVICE)

#  7. Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#  8. Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

#  9. Save the Model
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/pneumonia_model.pth")
print("Model saved to weights/pneumonia_model.pth")


# 10. Evaluation on Test Set
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nClassification Report:\n")
print(classification_report( y_true, y_pred,labels=[0, 1],target_names=["Normal", "Pneumonia"]))

