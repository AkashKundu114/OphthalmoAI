import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from prepare_dataset import prepare_dataloaders
from metric_logger import HardwareTelemetry

# RTX 5060 8GB Optimizations
torch.backends.cudnn.benchmark = True # Enable cuDNN auto-tuner

MANIFEST_PATH = './dataset/manifest.csv'
DATASET_ROOT = './dataset'
NUM_CLASSES = 12
BATCH_SIZE = 16 
EPOCHS = int(os.environ.get('EPOCHS', 1))

def get_convnext():
    model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

def get_densenet():
    model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

def get_efficientnet_v2():
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

class MetaEnsemble(nn.Module):
    def __init__(self, model1, model2, model3):
        super(MetaEnsemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        
        # Freeze base models
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False
        for param in self.model3.parameters():
            param.requires_grad = False
            
        # Meta-Classifier: takes logits from 3 models (12 * 3 = 36) and outputs 12
        self.meta_classifier = nn.Sequential(
            nn.Linear(NUM_CLASSES * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        with torch.no_grad():
            out1 = self.model1(x)
            out2 = self.model2(x)
            out3 = self.model3(x)
            
        stacked = torch.cat((out1, out2, out3), dim=1)
        return self.meta_classifier(stacked)

def train_base_model(model_name, model_fn, device, train_loader):
    print(f"--- Training Base Model: {model_name} ---")
    model = model_fn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(1, EPOCHS + 1):
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
                print(f"[{model_name}] Epoch [{epoch}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%")
                
    return model

def train():
    print("Initializing Meta-Classifier Ensemble Training...")
    telemetry = HardwareTelemetry(use_gpu=True, model_name="MetaEnsemble")
    
    train_loader, val_loader, test_loader, classes = prepare_dataloaders(
        MANIFEST_PATH, DATASET_ROOT, batch_size=BATCH_SIZE
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Phase 1: Train Base Models
    convnext = train_base_model("ConvNeXt-Small", get_convnext, device, train_loader)
    densenet = train_base_model("DenseNet-201", get_densenet, device, train_loader)
    effnet = train_base_model("EfficientNet-V2", get_efficientnet_v2, device, train_loader)
    
    # Save base models
    os.makedirs('models', exist_ok=True)
    torch.save(convnext.state_dict(), 'models/convnext_small.pth')
    torch.save(densenet.state_dict(), 'models/densenet201.pth')
    torch.save(effnet.state_dict(), 'models/efficientnet_v2_m.pth')
    print("Base models saved successfully.")
    
    # Phase 2: Train Meta Classifier
    print("--- Training Meta-Classifier ---")
    ensemble = MetaEnsemble(convnext, densenet, effnet).to(device)
    criterion = nn.CrossEntropyLoss()
    # Only optimizing meta_classifier parameters
    optimizer = optim.Adam(ensemble.meta_classifier.parameters(), lr=1e-3)
    
    for epoch in range(1, EPOCHS + 1):
        telemetry.start_epoch()
        ensemble.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            # Autocast for meta classifier
            with torch.amp.autocast('cuda'):
                outputs = ensemble(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward() # Scaler optional for simple linear layers, but works fine
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 10 == 0:
                current_acc = 100. * correct / total
                print(f"[Meta-Classifier] Epoch [{epoch}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%")
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        telemetry.end_epoch(epoch, epoch_loss, epoch_acc)

    print("Saving meta classifier weights...")
    torch.save(ensemble.meta_classifier.state_dict(), 'models/meta_classifier.pth')
    print("Training Complete & Ensemble Saved!")

if __name__ == '__main__':
    train()
