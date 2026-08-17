import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from prepare_dataset import prepare_dataloaders
from metric_logger import HardwareTelemetry

torch.backends.cudnn.benchmark = True

MANIFEST_PATH = './dataset/manifest.csv'
DATASET_ROOT = './dataset'
NUM_CLASSES = 12
BATCH_SIZE = 32
EPOCHS = 10

def get_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

def train():
    print("Initializing ResNet50 Training...")
    telemetry = HardwareTelemetry(use_gpu=True, model_name="ResNet50")
    
    train_loader, val_loader, test_loader, classes = prepare_dataloaders(
        MANIFEST_PATH, DATASET_ROOT, batch_size=BATCH_SIZE
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(1, EPOCHS + 1):
        telemetry.start_epoch()
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 10 == 0:
                current_acc = 100. * correct / total
                print(f"Epoch [{epoch}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%")
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        telemetry.end_epoch(epoch, epoch_loss, epoch_acc)

if __name__ == '__main__':
    train()



