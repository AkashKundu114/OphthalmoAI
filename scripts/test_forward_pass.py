import torch
from torchvision import models

def test_forward_pass():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    if device.type != 'cuda':
        print("CUDA is still not available.")
        return
        
    model = models.resnet50().to(device)
    model.eval()
    
    inputs = torch.randn(1, 3, 224, 224).to(device)
    
    print("Executing forward pass...")
    try:
        with torch.no_grad():
            outputs = model(inputs)
        print("Forward pass successful! The GPU is fully compatible and working.")
    except Exception as e:
        print(f"Crash during forward pass: {e}")

if __name__ == '__main__':
    test_forward_pass()
