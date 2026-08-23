import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from prepare_dataset import prepare_dataloaders
from metric_logger import HardwareTelemetry

torch.backends.cudnn.benchmark = False

MANIFEST_PATH = './dataset/manifest.csv'
DATASET_ROOT = './dataset'
NUM_CLASSES = 12
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
EPOCHS = int(os.environ.get('EPOCHS', 40))

def get_convnext(pretrained=True):
    weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
    model = models.convnext_small(weights=weights)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

def get_densenet(pretrained=True):
    weights = models.DenseNet201_Weights.DEFAULT if pretrained else None
    model = models.densenet201(weights=weights)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

def get_efficientnet_v2(pretrained=True):
    weights = models.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_v2_m(weights=weights)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

class MetaEnsemble(nn.Module):
    def __init__(self, model1, model2, model3):
        super(MetaEnsemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False
        for param in self.model3.parameters():
            param.requires_grad = False
            
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
    save_path = f"models/{model_name.lower().replace('-', '_')}_bf16.pth"
    
    # Auto-resume without redundant weight downloads
    if os.path.exists(save_path):
        print(f"\n=======================================================")
        print(f"⏩ Found existing checkpoint: {save_path}")
        print(f"Loading {model_name} from disk (skipping re-training)...")
        print(f"=======================================================")
        model = model_fn(pretrained=False)
        model.load_state_dict(torch.load(save_path, map_location='cpu'))
        return model

    print(f"\n=======================================================")
    print(f"--- Training Base Model: {model_name} (BF16, BS={BATCH_SIZE}) ---")
    print(f"=======================================================")
    
    model = model_fn(pretrained=True).to(device)
    telemetry = HardwareTelemetry(use_gpu=True, model_name=f"{model_name}_BF16_BS{BATCH_SIZE}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    total_steps = len(train_loader)
    for epoch in range(1, EPOCHS + 1):
        telemetry.start_epoch()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 10 == 0 or (i + 1) == total_steps:
                print(f"[{model_name}] Epoch [{epoch}/{EPOCHS}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%")
        
        epoch_loss = running_loss / total_steps
        epoch_acc = 100. * correct / total
        telemetry.end_epoch(epoch, epoch_loss, epoch_acc)
    
    # Save checkpoint immediately
    torch.save(model.state_dict(), save_path)
    print(f"✅ Checkpoint saved: {save_path}")
    
    # Safe cleanup & sync
    torch.cuda.synchronize()
    model.to('cpu')
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return model

def train():
    print(f"Initializing Meta-Classifier Ensemble Training [Precision: BF16, Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}]...")
    os.makedirs('models', exist_ok=True)
    
    train_loader, val_loader, test_loader, classes = prepare_dataloaders(
        MANIFEST_PATH, DATASET_ROOT, batch_size=BATCH_SIZE
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device} | Total Batches per Epoch: {len(train_loader)}")
    
    # 1. Train Base Models with individual telemetry logs & auto-resume
    convnext = train_base_model("ConvNeXt-Small", get_convnext, device, train_loader)
    densenet = train_base_model("DenseNet-201", get_densenet, device, train_loader)
    effnet = train_base_model("EfficientNet-V2-M", get_efficientnet_v2, device, train_loader)
    
    # 2. Train Meta-Classifier Head with dedicated telemetry log
    meta_path = 'models/meta_classifier_bf16.pth'
    if os.path.exists(meta_path):
        print(f"⏩ Meta-Classifier checkpoint already exists: {meta_path}")
        return

    print("\n=======================================================")
    print(f"--- Training Meta-Classifier Head (BF16, BS={BATCH_SIZE}) ---")
    print("=======================================================")
    meta_telemetry = HardwareTelemetry(use_gpu=True, model_name=f"MetaClassifier_BF16_BS{BATCH_SIZE}")
    ensemble = MetaEnsemble(convnext, densenet, effnet).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ensemble.meta_classifier.parameters(), lr=1e-3)
    
    total_steps = len(train_loader)
    for epoch in range(1, EPOCHS + 1):
        meta_telemetry.start_epoch()
        ensemble.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = ensemble(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 10 == 0 or (i + 1) == total_steps:
                current_acc = 100. * correct / total
                print(f"[Meta-Classifier] Epoch [{epoch}/{EPOCHS}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%")
            
        epoch_loss = running_loss / total_steps
        epoch_acc = 100. * correct / total
        meta_telemetry.end_epoch(epoch, epoch_loss, epoch_acc)

    print(f"\nSaving meta classifier weights to {meta_path}...")
    torch.save(ensemble.meta_classifier.state_dict(), meta_path)
    print(f"🎉 Training Complete! Individual Telemetry Logs & Checkpoints Saved for All 4 Models.")

if __name__ == '__main__':
    train()
