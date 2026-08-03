import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from sklearn.metrics import cohen_kappa_score, f1_score
import time
# Swin implements window self attention:
# divides the image feature map into non-overlapping windows
# within which attention is calculated
# gen_fullres stitches these windows into a full res grid

def gen_fullres(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    num_windows_hw = (H // window_size) * (W // window_size)
    B = windows.shape[0] // num_windows_hw
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size)
    x = x.permute(0, 1, 3, 2, 4).contiguous().view(B, H, W)
    return x


class MSDF(nn.Module):
    def __init__(self, in_channels=(192, 384, 768), common_dim=384, gate_hidden=128):
        super().__init__()
        self.num_scales = len(in_channels)
        self.proj = nn.ModuleList([nn.Conv2d(c, common_dim, 1) for c in in_channels])
        self.proj_norm = nn.ModuleList([nn.BatchNorm2d(common_dim) for _ in in_channels])

        self.gate = nn.Sequential(
            nn.Linear(common_dim * self.num_scales, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.num_scales),
        )

        self.refine_dw = nn.Conv2d(common_dim, common_dim, 3, padding=1, groups=common_dim, bias=False)
        self.refine_pw = nn.Conv2d(common_dim, common_dim, 1, bias=False)
        self.refine_bn = nn.BatchNorm2d(common_dim)
        self.act = nn.GELU()

        self.out_dim = common_dim

    def forward(self, feats):
        target_hw = max(f.shape[-1] for f in feats)
        projected = []
        pooled = []
        for f, proj, norm in zip(feats, self.proj, self.proj_norm):
            p = self.act(norm(proj(f)))
            if p.shape[-1] != target_hw:
                p = F.interpolate(p, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
            projected.append(p)
            pooled.append(F.adaptive_avg_pool2d(p, 1).flatten(1))

        context = torch.cat(pooled, dim=1)
        gate_logits = self.gate(context)
        gates = torch.softmax(gate_logits, dim=1)

        fused = sum(gates[:, i].view(-1, 1, 1, 1) * projected[i] for i in range(self.num_scales))

        refined = self.refine_pw(self.refine_dw(fused))
        refined = self.act(self.refine_bn(refined))
        fused = fused + refined

        return fused, gates


class MDSFSwinTeacher(nn.Module):
    def __init__(self, num_classes=10, pretrained_backbone=True, fusion_common_dim=576, fusion_gate_hidden=192):
        super().__init__()
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained_backbone, num_classes=0)
        stage_channels = [96, 192, 384, 768]
        self.fusion = MSDF(
            in_channels=stage_channels[1:],
            common_dim=fusion_common_dim,
            gate_hidden=fusion_gate_hidden,
        )
        self.classifier = nn.Linear(self.fusion.out_dim, num_classes)

        self._raw_attn = {}
        self._stage_meta = {}
        self._hook_handles = []
        self._register_attention_hooks()
        self._stage_feats = {}
        self._register_feature_hooks()

    def _register_attention_hooks(self):
        for stage_idx, layer in enumerate(self.backbone.layers):
            block = layer.blocks[0]
            def make_hook(stage_idx=stage_idx, block=block):
                def hook(module, inp, out):
                    self._raw_attn[stage_idx] = out.detach()
                    self._stage_meta[stage_idx] = (block.window_size[0], *block.input_resolution)
                return hook
            self._hook_handles.append(block.attn.softmax.register_forward_hook(make_hook()))

    def _attn_to_saliency(self, stage_idx):
        attn = self._raw_attn[stage_idx]
        ws, H, W = self._stage_meta[stage_idx]
        saliency = attn.sum(dim=2).mean(dim=1)
        saliency = saliency.view(-1, ws, ws)
        saliency = gen_fullres(saliency, ws, H, W)
        return saliency.unsqueeze(1)

    def _register_feature_hooks(self):
        for stage_idx, layer in enumerate(self.backbone.layers):
            def make_hook(stage_idx=stage_idx):
                def hook(module, inp, out):
                    feat = out
                    if feat.dim() == 3:
                        H, W = layer.blocks[0].input_resolution
                        feat = feat.view(feat.shape[0], H, W, -1)
                    feat = feat.permute(0, 3, 1, 2).contiguous()
                    self._stage_feats[stage_idx] = feat
                return hook
            self._hook_handles.append(layer.register_forward_hook(make_hook()))

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def forward(self, x):
        self._raw_attn.clear()
        self._stage_feats.clear()

        _ = self.backbone.forward_features(x)

        feats = [self._stage_feats[i] for i in (1, 2, 3)]
        fused, gates = self.fusion(feats)
        pooled = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        logits = self.classifier(pooled)

        saliency_maps = [self._attn_to_saliency(i) for i in range(4)]
        return logits, saliency_maps, feats, fused, gates


def selftest():
    torch.manual_seed(0)
    model = MDSFSwinTeacher(num_classes=10, pretrained_backbone=False)

    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    head_params = sum(p.numel() for p in model.classifier.parameters())
    total_params = backbone_params + fusion_params + head_params

    print(f"Swin-Tiny backbone params: {backbone_params/1e6:.3f}M")
    print(f"Fusion head params: {fusion_params/1e6:.3f}M")
    print(f"Classifier head params: {head_params/1e6:.3f}M")
    print(f"Total: {total_params/1e6:.3f}M")

    x = torch.randn(2, 3, 224, 224)
    logits, saliency_maps, feats, fused, gates = model(x)
    print("logits:", logits.shape)
    for i, m in enumerate(saliency_maps):
        print(f"stage {i} saliency map: {tuple(m.shape)}")

    loss = logits.sum()
    loss.backward()
    print("Backward pass OK, gradients flow through backbone + fusion + head.")


def train(epochs=5, batch_size=32, lr=1e-4, data_root="./data", print_every=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    start_time = time.time()

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

    model = MDSFSwinTeacher(num_classes=10, pretrained_backbone=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    #variable for recording best accuracy
    best_accuracy = 0
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running = 0.0
        num_batches = len(train_loader)
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, _, _, _, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            running += loss.item()

            if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == num_batches:
                print(f"epoch {epoch+1}/{epochs} batch {batch_idx+1}/{num_batches} "
                      f"loss={running/(batch_idx+1):.4f}")

        model.eval()
        correct, total = 0, 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():

            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _, _, _, _ = model(x)
                predictions = logits.argmax(1)
                correct += (predictions == y).sum().item()
                total += y.size(0)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
            accuracy = correct / total
            kappa = cohen_kappa_score(all_labels, all_predictions)
            macro_F1 = f1_score(all_labels, all_predictions, average="macro")

        #record total training time and output results
        end_time = time.time()
        training_time = end_time - start_time
        print(f"\nTraining time: {training_time/60:.2f} Minutes")
        #output results per epoch
        print(f"Epoch: {epoch + 1}     Loss: {running/len(train_loader):.4f}")
        print(f"Test Accuracy: {accuracy*100:.2f}%     Kappa Score: {kappa:.4f}     Macro-F1 Score: {macro_F1:.4f}")

        #check if current epoch is best so far and save model weights if so
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "mdsf_swin_teacher_trained_EuroSAT.pt")
            print("Saved mdsf_swin_teacher_trained_EuroSAT.pt")

    return model

if __name__ == "__main__":
    selftest()
    train(epochs=10)