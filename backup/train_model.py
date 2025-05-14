import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
import numpy as np

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

# Selected classes with emojis
CLASS_EMOJIS = {
    'apple': '🍎', 'aquarium_fish': '🐠', 'baby': '👶', 'bear': '🐻', 
    'beaver': '🦫', 'bed': '🛏️', 'bee': '🐝', 'beetle': '🪲',
    'bicycle': '🚲', 'bottle': '🍾', 'bowl': '🥣', 'boy': '👦', 
    'bridge': '🌉', 'bus': '🚌', 'butterfly': '🦋', 'camel': '🐪',
    'can': '🥫', 'castle': '🏰', 'caterpillar': '🐛', 'cattle': '🐄', 
    'chair': '🪑', 'chimpanzee': '🦧', 'clock': '⏰', 'cloud': '☁️',
    'cockroach': '🪳', 'couch': '🛋️', 'crab': '🦀', 'crocodile': '🐊', 
    'cup': '☕', 'dinosaur': '🦖', 'dolphin': '🐬', 'elephant': '🐘',
    'flatfish': '🐟', 'rose': '🌹', 'fox': '🦊', 'girl': '👧', 
    'hamster': '🐹', 'house': '🏠', 'kangaroo': '🦘', 'keyboard': '⌨️',
    'lamp': '💡', 'lawn_mower': '🚜', 'leopard': '🐆', 'lion': '🦁', 
    'lizard': '🦎', 'lobster': '🦞', 'man': '👨', 'maple_tree': '🍁',
    'mountain': '⛰️', 'mouse': '🐁'
}

SELECTED_CLASSES = list(CLASS_EMOJIS.keys())

def train_model():
    print("\nSetting up training...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transforms with augmentation for better accuracy
    train_transform = transforms.Compose([
        transforms.Resize(224),  # Back to 224x224 for better accuracy
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])
    
    print("\nLoading CIFAR-100 dataset...")
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    print("\nPreparing class mappings...")
    trainset.targets = torch.LongTensor(trainset.targets)
    testset.targets = torch.LongTensor(testset.targets)
    
    cifar100_to_idx = {cls: idx for idx, cls in enumerate(trainset.classes)}
    selected_indices = [cifar100_to_idx[cls] for cls in SELECTED_CLASSES]
    
    print("\nFiltering dataset...")
    train_mask = torch.isin(trainset.targets, torch.tensor(selected_indices))
    test_mask = torch.isin(testset.targets, torch.tensor(selected_indices))
    
    train_indices = torch.where(train_mask)[0].tolist()
    test_indices = torch.where(test_mask)[0].tolist()
    
    print(f"Selected {len(train_indices)} training samples")
    print(f"Selected {len(test_indices)} test samples")
    
    trainloader = DataLoader(
        Subset(trainset, train_indices),
        batch_size=32,  # Smaller batch size for better generalization
        shuffle=True,
        num_workers=0
    )
    
    testloader = DataLoader(
        Subset(testset, test_indices),
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    # Use ResNet50 for higher accuracy
    print("\nLoading pre-trained ResNet50...")
    model = models.resnet50(weights='IMAGENET1K_V2')
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Add dropout for regularization
        nn.Linear(num_ftrs, len(SELECTED_CLASSES))
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    
    print("\nStarting training...")
    best_acc = 0
    patience = 0
    max_patience = 5
    
    for epoch in range(20):  # More epochs for higher accuracy
        print(f"\nEpoch {epoch+1}/20")
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(trainloader):
            # Map labels to new indices
            new_labels = torch.tensor([
                selected_indices.index(label.item())
                for label in labels
            ])
            
            inputs = inputs.to(device)
            new_labels = new_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, new_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
        
        print("\nEvaluating...")
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                new_labels = torch.tensor([
                    selected_indices.index(label.item())
                    for label in labels
                ])
                
                inputs = inputs.to(device)
                new_labels = new_labels.to(device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += new_labels.size(0)
                correct += predicted.eq(new_labels).sum().item()
        
        accuracy = 100. * correct / total
        print(f'Accuracy: {accuracy:.1f}%')
        
        # Learning rate scheduling
        scheduler.step(accuracy)
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'model.pth')
            print(f'New best model saved with accuracy: {accuracy:.1f}%')
            patience = 0
        else:
            patience += 1
            
        # Early stopping with patience
        if patience >= max_patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break
            
        # Stop if we reach target accuracy
        if accuracy >= 94.0:
            print(f"\nReached target accuracy of 94%!")
            break
    
    print(f'\nTraining finished! Best accuracy: {best_acc:.1f}%')

if __name__ == '__main__':
    train_model()
