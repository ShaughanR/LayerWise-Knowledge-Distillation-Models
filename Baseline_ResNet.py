import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import teacher_model
import torchvision.models as models
from torch.optim import optimizer
from torchvision.datasets import ImageFolder
import time
from sklearn.metrics import cohen_kappa_score, f1_score

class ResNetBaseline(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        # if-else for determining if model should be pretrained or not on ImageNet weights
        if pretrained == True:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, images):
        return self.model(images)

def Train(epochs, learning_rate, batch_size, pretrain):
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # AID RGB stats
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full = ImageFolder(root="./archive_AID/AID", transform=tf)

    n_val = int(0.2 * len(full))
    train_set, val_set = random_split(full, [len(full) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    print(f"Split: {len(train_set)} train / {len(val_set)} val.")


    #initialize model
    resnet = ResNetBaseline(num_classes=30, pretrained=pretrain).to(device)
    #initialize optimizer
    optimizer = torch.optim.AdamW(resnet.parameters(), learning_rate, weight_decay=0.0001)

    #get number of parameters
    resnet_params = sum(p.numel() for p in resnet.parameters())
    print(f"Student Model Parameters: {resnet_params / 1e6:.2f}M")

    for epoch in range(epochs):
        resnet.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = resnet(images)
            loss = F.cross_entropy(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        accuracy, kappa, macro_F1 = Test(resnet, val_loader, device)

        #record total training time and output results
        end_time = time.time()
        training_time = end_time - start_time
        print(f"\nTraining time: {training_time/60:.2f} Minutes")
        #output results per epoch
        print(f"Epoch: {epoch + 1}     Loss: {total_loss/len(train_loader):.4f}")
        print(f"Test Accuracy: {accuracy*100:.2f}%     Kappa Score: {kappa:.4f}     Macro-F1 Score: {macro_F1:.4f}")

def Test(model, val_loader, device):
    model.eval()

    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()

            total  += labels.size(0)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # compute accuracy
    accuracy = correct / total
    # compute kappa score
    kappa = cohen_kappa_score(all_labels, all_predictions)
    # compute f1 score
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")
    return accuracy, kappa, macro_f1

Train(10, 0.0005, 32, True)