import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
import argparse

# Define transfer learning model
class BrainTumorResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorResNet, self).__init__()
        # Load pre-trained ResNet18
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Freeze convolutional layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Data preprocessing and augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = ImageFolder('/Users/mac/Desktop/CV A1/breast/projetFini/compilationFini/codetest/data/brain_tumor/training', transform=train_transform)
test_dataset = ImageFolder('/Users/mac/Desktop/CV A1/breast/projetFini/compilationFini/codetest/data/brain_tumor/testing', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# Evaluation function
def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total}%')

# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Brain Tumor CNN with Transfer Learning')
    parser.add_argument('--model', choices=['pytorch', 'tensorflow'], default='pytorch')
    args = parser.parse_args()

    if args.model == 'pytorch':
        model = BrainTumorResNet()
        criterion = nn.CrossEntropyLoss()
        # Optimize only the final layer
        optimizer = optim.Adam(model.resnet.fc.parameters(), lr=0.001)
        train_model(model, train_loader, criterion, optimizer)
        evaluate_model(model, test_loader)
        torch.save(model.state_dict(), 'Victor_model.torch')
        print("PyTorch model saved as Victor_model.torch")