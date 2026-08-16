import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

class RetinalDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform
        
        # Create a mapping from class name to label index
        self.classes = sorted(self.data_frame['class'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_frame.iloc[idx, 0].replace('\\', '/')
        img_name = os.path.join(self.root_dir, img_path)
        image = Image.open(img_name).convert('RGB')
        label = self.class_to_idx[self.data_frame.iloc[idx, 1]]

        if self.transform:
            image = self.transform(image)

        return image, label

def prepare_dataloaders(csv_file, root_dir, batch_size=16, test_size=0.2, val_size=0.1):
    df = pd.read_csv(csv_file)
    
    # Split the dataset
    train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), stratify=df['class'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=(test_size / (test_size + val_size)), stratify=temp_df['class'], random_state=42)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = RetinalDataset(train_df, root_dir, transform=train_transform)
    val_dataset = RetinalDataset(val_df, root_dir, transform=val_test_transform)
    test_dataset = RetinalDataset(test_df, root_dir, transform=val_test_transform)
    
    # DataLoaders for 8GB VRAM (batch_size=16 or 32 is safe)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Prepared DataLoaders:")
    print(f"Training: {len(train_dataset)} images")
    print(f"Validation: {len(val_dataset)} images")
    print(f"Testing: {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx

if __name__ == '__main__':
    manifest_path = './dataset/manifest.csv'
    dataset_root = './dataset'
    
    # Safe batch size for RTX 5060 8GB
    BATCH_SIZE = 16 
    
    train_loader, val_loader, test_loader, classes = prepare_dataloaders(manifest_path, dataset_root, batch_size=BATCH_SIZE)
    print("Classes mapped to labels:", classes)
    print("Data is ready for training on RTX 5060 8GB!")

