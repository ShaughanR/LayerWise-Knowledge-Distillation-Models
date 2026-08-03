import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import time
from sklearn.metrics import cohen_kappa_score, f1_score
import teacher_model

class ResNetStudent(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        # if-else for determining if model should be pretrained or not on ImageNet weights
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        # initialize resnet model
        model = models.resnet18(weights=weights)
        # stem layer for resnet
        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        # resnet layers that line up with teacher stages
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        # pool and flatten after
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    # forward pass for training resnet model
    def forward(self, images):
        # converts inputted images into features maps
        images = self.stem(images)
        # save the result of each layer in a separate variable
        images = self.layer1(images)
        l1 = self.layer2(images)
        l2 = self.layer3(l1)
        l3 = self.layer4(l2)
        # flatten result of final layer
        pooled = self.pool(l3).flatten(1)
        # get logits from pooled features
        logits = self.classifier(pooled)

        # return result of final layer as well as each individual layer for layer-wise KD
        return {
            "logits": logits,
            "feature maps": [l1, l2, l3]
        }

class StudentTeacherProjector(nn.Module):
    def __init__(self):
        super().__init__()
        # used to conver student channel sizes to teacher sizes
        self.projectors = nn.ModuleList([
            nn.Conv2d(128, 192, 1),
            nn.Conv2d(256, 384, 1),
            nn.Conv2d(512, 768, 1),
        ])
    # forward pass to convert between student and teacher sizes
    # student holds all returned values from student model's forward pass
    def forward(self, student):
        projected = []
        # converts feature maps into teacher stage's scales
        for proj, feat in zip(self.projectors, student["feature maps"]):
            # append scaled feature maps to list
            projected.append(proj(feat))
        return projected

# function used to compute kd-loss
def kd_loss(student_logits, teacher_logits, temperature=4.0):
    # get softmax score using only students values
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    # get softmax score using only teacher values
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    # compute kl-divergence value using student and teacher softmax values
    loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
    # multiple kl-divergence score by temperature to get final loss
    loss *= temperature ** 2
    return loss

# function used to compute total loss for all knowledge distillation layers
def ComputeLoss(student, teacher, labels, projector):
    # call projector on student so values are comparable to teacher model
    projected = projector(student)
    loss_mse = sum(F.mse_loss(student, teacher) for student, teacher in zip(projected, teacher[2])) / len(projected)
    # compute kd loss between student and teacher
    kd_loss_value = kd_loss(student["logits"], teacher[0])
    # compute loss using only students values
    loss_ce = F.cross_entropy(student["logits"], labels)
    # get total weighted loss from all loss calculations
    total_loss = (loss_ce + 0.25 * kd_loss_value + 0.25 * loss_mse)
    # print(f"loss ce {loss_ce.item()}")
    # print(f"loss kd {loss_logits.item()}")
    # print(f"loss mse {loss_mse.item()}")
    return total_loss

def Train(epochs, batch_size, learning_rate, pretrained):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # set up tensors for normalization
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # AID RGB stats
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # get data
    full = ImageFolder(root="./archive_AID/AID", transform=tf)
    # 80/20 split
    n_val = int(0.2 * len(full))
    # split data
    train_set, val_set = random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))
    # load data loader for train and test
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    print(f"Split: {len(train_set)} train / {len(val_set)} val.")

    # initialize teacher model
    teacher = teacher_model.MDSFSwinTeacher(num_classes=30, pretrained_backbone=False).to(device)
    # fill teacher model's weights with saved weights
    teacher.load_state_dict(torch.load("mdsf_swin_teacher_trained_AID (1).pt"))
    # set teacher model to eval mode
    teacher.eval()
    # freeze teacher model weights for knowledge distillation
    for p in teacher.parameters():
        p.requires_grad = False

    # initialize student model
    student = ResNetStudent(num_classes=30, pretrained=pretrained).to(device)
    # initialize projector for teacher-student comparisons
    projector = StudentTeacherProjector().to(device)
    # initialize optimizer
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(projector.parameters()),
                                  lr=learning_rate, weight_decay=0.0001)

    # get number of parameters for student model and projector
    student_params = sum(p.numel() for p in student.parameters())
    projector_params = sum(p.numel() for p in projector.parameters())
    print(f"Student Model Parameters: {student_params / 1e6:.2f}M")
    print(f"Projector Parameters: {projector_params / 1e6:.2f}M")
    print(f"Total Parameters: {(student_params + projector_params) / 1e6:.2f}M")

    #start time for tracking training time
    start_time = time.time()
    # iterate through epochs
    for epoch in range(epochs):
        # set student and projector models to training mode
        student.train()
        projector.train()
        # variable for holding loss between batches
        running_loss = 0.0
        # iterate through data batches
        for images, labels in train_loader:
            # make sure gpu is being used
            images = images.to(device)
            labels = labels.to(device)
            # get outputted values from teacher forward pass
            with torch.no_grad():
                teacher_output = teacher(images)
            # get outputted values from student forward pass
            student_output = student(images)
            # compute loss values using knowledge distillation between teacher and student
            loss = ComputeLoss(student_output, teacher_output, labels, projector)
            # clear gradient from previous batch
            optimizer.zero_grad()
            # compute loss values for each layer in student model
            loss.backward()
            # update loss values in student layers
            optimizer.step()
            # compute combined loss for all batches
            running_loss += loss.item()
        # compute accuracy, kappa, and macro_f1 score per epoch using test method
        accuracy, kappa, macro_f1 = Test(student, val_loader, device)

        # record total training time and output results
        end_time = time.time()
        training_time = end_time - start_time
        print(f"\nTraining time: {training_time/60:.2f} Minutes")
        # output results per epoch
        print(f"Epoch: {epoch + 1}     Loss: {running_loss/len(train_loader):.4f}")
        print(f"Test Accuracy: {accuracy*100:.2f}%     Kappa Score: {kappa:.4f}     Macro-F1 Score: {macro_f1:.4f}")


# testing method to record accuracy of model
def Test(student, val_loader, device):
    # set model to eval mode
    student.eval()
    # variables for recording accuracy and loss
    correct = 0
    total = 0
    # lists for holding total labels and predictions
    all_labels = []
    all_predictions = []

    # freeze model weights
    with torch.no_grad():
        # iterate through batches
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            # get student output for current batch
            outputs = student(images)
            # get prediction values
            predictions = outputs["logits"].argmax(dim=1)
            # compute number of correct predictions
            correct += (predictions == labels).sum().item()
            # total up number of predictions made
            total += labels.size(0)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # compute accuracy
    accuracy = correct / total
    # compute kappa score
    kappa = cohen_kappa_score(all_labels, all_predictions)
    # compute f1 score
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")
    return accuracy, kappa, macro_f1

# call train method
Train(10, 32, 0.0005, True)