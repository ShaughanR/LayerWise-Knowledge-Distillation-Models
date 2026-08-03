import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import time
from sklearn.metrics import cohen_kappa_score, f1_score
import teacher_model


class MSDFResNetStudent(nn.Module):
    def __init__(self,num_classes=10, pretrained=True):
        super().__init__()

        #if-else for determining if model should be pretrained or not on ImageNet weights
        if pretrained == True:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        #initialize resnet backbone of student model
        student_model = models.resnet18(weights=weights)

        #stem layer for resnet
        self.stem = nn.Sequential(
            student_model.conv1,
            student_model.bn1,
            student_model.relu,
            student_model.maxpool,
        )

        #resnet blocks that line up with teacher stages
        self.layer1 = student_model.layer1
        self.layer2 = student_model.layer2
        self.layer3 = student_model.layer3
        self.layer4 = student_model.layer4

        #MSDF module for student for stronger knowledge distillation
        self.msdf = teacher_model.MSDF(
            in_channels=(128, 256, 512),
            common_dim=256,
            gate_hidden=64
        )

        #gives class predictions from MSDF output
        self.classifier = nn.Linear(
            self.msdf.out_dim,
            num_classes,
        )

    #forward pass for training resnet model
    def forward(self, images):
        #converts inputted images into features maps
        images = self.stem(images)
        #four resnet blocks corresponding to four teacher blocks
        images = self.layer1(images)
        l1 = self.layer2(images)
        l2 = self.layer3(l1)
        l3 = self.layer4(l2)

        #get fused features and weighting gates from resnet features
        fused, gates = self.msdf([l1, l2, l3])
        #pooling to vectorize fused features
        pooled = F.adaptive_avg_pool2d(fused,1,).flatten(1)
        #get logits from pooled features
        logits = self.classifier(pooled)

        #returns for student model forward pass
        #logits: used for loss calculations
        #stage_feats: used for knowledge distillation in student model's layers
        #fused_feat: used for knowledge distillation between student and teacher MSDF outputs
        #gates: used for knowledge distillation between student and teacher MSDF outputs
        return {
            "logits": logits,
            "stage_feats": [l1, l2, l3],
            "fused_feat": fused,
            "gates": gates,
        }

class TeacherStudentProjector(nn.Module):
    def __init__(self):
        super().__init__()

        #used to conver student channel sizes to teacher sizes
        self.stage = nn.ModuleList([
            nn.Conv2d(128, 192, 1),
            nn.Conv2d(256, 384, 1),
            nn.Conv2d(512, 768, 1),
        ])

        #converts student MSDF feature sizes to teacher sizes
        self.fusion = nn.Conv2d(256,576,1)


    #forward pass to convert between student and teacher sizes
    #student holds all returned values from student model's forward pass
    def forward(self, student):
        #used to hold transformed features from student model
        stages = []

        #combine student stage features with self.stage stages as tuples
        for stage, feat in zip(self.stage,student["stage_feats"]):
            #converts stage_feats into self.stage's scales and appends them to stages
            stages.append(stage(feat))

        #change student fused feature size to teacher size
        fused = self.fusion(student["fused_feat"])
        #returns student values projected to teacher size
        return stages, fused

#function used to compute kd-loss
def kd_loss(student_logits, teacher_logits, temperature = 4.0):
    student_softmax = F.log_softmax(student_logits / temperature, dim=1)
    teacher_softmax = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(student_softmax, teacher_softmax, reduction="batchmean")
    loss *= temperature ** 2
    return loss

#function used to compute total loss for all knowledge distillation areas
def ComputeLoss(student, teacher, labels, projector):

    #convert student outputs to teacher output scale
    projector_stages, projector_fused = projector(student)

    #Stage feature distillation computed from projected student layers and teacher stages
    loss_stage = sum(F.mse_loss(s,t) for s,t in zip(projector_stages, teacher[2]))

    #calculate all loss values for layers and MSDF output
    loss_fusion = F.mse_loss(projector_fused, teacher[3])
    loss_gate = F.kl_div(student["gates"].log(), teacher[4], reduction="batchmean")
    loss_kd = kd_loss(student["logits"], teacher[0])
    loss_ce = F.cross_entropy(student["logits"], labels)
    total_loss = (loss_ce + loss_kd + 5 * loss_stage + 3 * loss_fusion + 0.5 * loss_gate)


    return total_loss

def Train(epochs, batch_size, learning_rate, pretrain):
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

    #initialize teacher model
    teacher = teacher_model.MDSFSwinTeacher(num_classes=45, pretrained_backbone=False).to(device)
    #fill teacher model's weights with saved weights
    teacher.load_state_dict(torch.load("mdsf_swin_teacher_trained_AID.pt"))
    #set teacher model to eval mode
    teacher.eval()
    #freeze teacher weights for knowledge distillation
    for param in teacher.parameters():
        param.requires_grad = False

    #initialize student model
    student = MSDFResNetStudent(num_classes=45, pretrained=pretrain).to(device)
    #initialize projector for teacher-student comparisons
    projector = TeacherStudentProjector().to(device)
    #initialize optimizer
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(projector.parameters()), learning_rate, weight_decay=0.0001)

    #start time for tracking training time
    start_time = time.time()
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        #set student and projector models to training mode
        student.train()
        projector.train()
        #variable for holding loss between batches
        running_loss = 0

        #iterate through data batches
        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                teacher_output = teacher(images)

            #get outputted values from student forward pass
            student_output = student(images)

            #compute loss values using knowledge distillation between teacher and student
            total_loss = ComputeLoss(student_output, teacher_output, labels, projector)
            #clear gradient from previous batch
            optimizer.zero_grad()
            #compute loss values for each layer in student model
            total_loss.backward()
            #update loss values in student layers
            optimizer.step()
            #compute combined loss for all batches
            running_loss += total_loss.item()
        #call test_accuracy on student model to get accuracy for current epoch
        accuracy, kappa, macro_f1 = Test(student, val_loader, device)

        # record total training time and output results
        end_time = time.time()
        training_time = end_time - start_time
        print(f"\nTraining time: {training_time / 60:.2f} Minutes")
        # output results per epoch
        print(f"Epoch: {epoch + 1}     Loss: {running_loss / len(train_loader):.4f}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%     Kappa Score: {kappa:.4f}     Macro-F1 Score: {macro_f1:.4f}")

def Test(student, val_loader, device):
    # set student to eval mode
    student.eval()
    #variables for recording accuracy and loss
    correct = 0
    total = 0
    running_loss = 0.0
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

#call train method
Train(5, 32, 0.0001, pretrain=True)
