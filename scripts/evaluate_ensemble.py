import os
import torch
import torch.nn as nn
from torchvision import models
from prepare_dataset import prepare_dataloaders
from sklearn.metrics import classification_report, accuracy_score

MANIFEST_PATH = './dataset/manifest.csv'
DATASET_ROOT = './dataset'
NUM_CLASSES = 12

def get_convnext():
    model = models.convnext_small(weights=None)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

def get_densenet():
    model = models.densenet201(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

def get_efficientnet_v2():
    model = models.efficientnet_v2_m(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

class MetaEnsemble(nn.Module):
    def __init__(self, model1, model2, model3):
        super(MetaEnsemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
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

def eval_model(model, name, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
    
    acc = accuracy_score(all_targets, all_preds) * 100
    print(f"\n>>> [{name}] Test Accuracy: {acc:.2f}%")
    return acc, all_targets, all_preds

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating models on: {device}")
    
    _, _, test_loader, classes = prepare_dataloaders(MANIFEST_PATH, DATASET_ROOT, batch_size=16)
    target_names = [k for k, v in sorted(classes.items(), key=lambda item: item[1])]
    
    # 1. Load ConvNeXt
    convnext = get_convnext().to(device)
    if os.path.exists('models/convnext_small.pth'):
        convnext.load_state_dict(torch.load('models/convnext_small.pth', map_location=device))
        eval_model(convnext, "ConvNeXt-Small", test_loader, device)

    # 2. Load DenseNet-201
    densenet = get_densenet().to(device)
    if os.path.exists('models/densenet201.pth'):
        densenet.load_state_dict(torch.load('models/densenet201.pth', map_location=device))
        eval_model(densenet, "DenseNet-201", test_loader, device)

    # 3. Load EfficientNet-V2-M
    effnet = get_efficientnet_v2().to(device)
    if os.path.exists('models/efficientnet_v2_m.pth'):
        effnet.load_state_dict(torch.load('models/efficientnet_v2_m.pth', map_location=device))
        eval_model(effnet, "EfficientNet-V2-M", test_loader, device)

    # 4. Load Meta Ensemble
    if os.path.exists('models/meta_classifier.pth'):
        ensemble = MetaEnsemble(convnext, densenet, effnet).to(device)
        ensemble.meta_classifier.load_state_dict(torch.load('models/meta_classifier.pth', map_location=device))
        acc, targets, preds = eval_model(ensemble, "MetaEnsemble (Final)", test_loader, device)
        print("\n=== Final MetaEnsemble Classification Report ===")
        print(classification_report(targets, preds, target_names=target_names, digits=4))

if __name__ == '__main__':
    main()
