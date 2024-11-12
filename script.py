import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import timm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix

# Custom Dataset Class to fetch data
class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

# Transform
transform = transforms.Compose([
    transforms.Resize((408, 408)),  # Adjusted for the model requirements
    transforms.ToTensor(),
])

# Data Loaders
batch_size = 8

train_dataset = CustomDataset(csv_file='train.csv', img_dir='train_resized', transform=transform)
val_dataset = CustomDataset(csv_file='val.csv', img_dir='val_resized', transform=transform)
test_dataset = CustomDataset(csv_file='test.csv', img_dir='test_resized', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
model = timm.create_model('crossvit_18_dagger_408', pretrained=True, num_classes=1)
model = model.to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Calculate Sensitivity and Specificity 
def calculate_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

# Training and Validation Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_val_acc = 0.0
    best_val_auc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.round(torch.sigmoid(outputs))
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / len(train_loader.dataset)

        val_loss, val_acc, val_auc, val_sensitivity, val_specificity = evaluate_model(model, val_loader, criterion)

        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}')
        print(f'Validation Sensitivity: {val_sensitivity:.4f} Specificity: {val_specificity:.4f}')

        if val_auc > best_val_auc:
            best_val_acc = val_acc
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_model_crossvit.pth')

    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    print(f'Best Validation AUC: {best_val_auc:.4f}')

def evaluate_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.round(torch.sigmoid(outputs))
            correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / len(data_loader.dataset)
    epoch_auc = roc_auc_score(all_labels, all_outputs)
    
    # Calculate sensitivity and specificity
    sensitivity, specificity = calculate_sensitivity_specificity(all_labels, [1 if x >= 0.5 else 0 for x in all_outputs])

    return epoch_loss, epoch_acc, epoch_auc, sensitivity, specificity

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)       # Adjust the number of epochs

# Load the best model and evaluate on the test set
model.load_state_dict(torch.load('best_model_crossvit.pth'))
test_loss, test_acc, test_auc, test_sensitivity, test_specificity = evaluate_model(model, test_loader, criterion)

print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f}')
print(f'Test Sensitivity: {test_sensitivity:.4f} Specificity: {test_specificity:.4f}')