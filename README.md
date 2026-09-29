# Efficient Satellite Image Classification via Knowledge Distillation

A PyTorch-based investigation of knowledge distillation techniques for compressing a high-performance Multi-Scale Dynamic Fusion (MSDF) Swin Transformer into lightweight ResNet-18 student models for remote-sensing image classification.

The project evaluates whether smaller convolutional models can retain the classification performance of a substantially larger transformer while reducing model size and training cost.

## Key Results

The strongest student model used layer-wise knowledge distillation and achieved:

| Model | EuroSAT Accuracy | Parameters |
| --- | ---: | ---: |
| MSDF Swin Transformer Teacher | 98.50% | 28.975M |
| Layer-Wise KD ResNet-18 | **98.04%** | **11.71M |
| Baseline ResNet-18 | 97.09% | ~11M |

The layer-wise student retained nearly all of the teacher's EuroSAT classification accuracy while using roughly 40% as many parameters.

Across the experiments, knowledge-distilled students were evaluated on:
- EuroSAT
- NWPU-RESISC45
- AID

The project also compared multiple distillation strategies, including:
- Layer-wise distillation
- Attention-map distillation
- Multi-teacher distillation
- Adversarial distillation
- Graph-based distillation

## Motivation

High-performing vision transformer architectures can achieve excellent remote-sensing classification accuracy, but their computational requirements can limit their usefulness in resource-constrained environments.

This project explores whether knowledge distillation can transfer useful representations from an MSDF Swin Transformer teacher to smaller ResNet-18 students while preserving classification performance.

The primary goals were to:

- Reduce model parameter count
- Reduce training cost
- Preserve classification accuracy
- Compare different knowledge-distillation strategies
- Evaluate performance across remote-sensing datasets with different levels of complexity

## Architecture

### Teacher Model

The teacher uses a Swin-Tiny backbone with a Multi-Scale Dynamic Fusion (MSDF) module.

The MSDF module:
1. Extracts intermediate Swin feature maps at multiple scales
2. Projects them into a shared feature dimension
3. Learns dynamic gating weights for each scale
4. Combines the weighted representations
5. Produces a fused feature representation for classification

The teacher also exposes intermediate feature maps and attention information for use during knowledge distillation.

### Student Model

The primary student architecture is based on ResNet-18.

Intermediate ResNet feature maps are extracted from multiple residual stages and aligned with corresponding teacher representations through learned 1x1 convolutional projection layers.

### Layer-Wise Knowledge Distillation

The strongest student was trained using a combination of:

- **Cross-Entropy Loss** — learns from ground-truth class labels
- **KL-Divergence Loss** — transfers the teacher's softened class probability distribution
- **Feature MSE Loss** — aligns intermediate student feature representations with teacher feature maps

The layer-wise loss combines both final model predictions and intermediate representations, allowing the student to learn more information than response-based distillation alone.

## Knowledge Distillation Pipeline

                    Input Image
                        |
            +-----------+-----------+
            |                       |
            v                       v
     MSDF Swin Teacher        ResNet-18 Student
            |                       |
     Teacher Logits            Student Logits
            |                       |
            +------ KL Loss --------+
            |
     Teacher Feature Maps
            |
        Projection
            |
     Student Feature Maps
            |
          MSE Loss

Ground Truth -> Cross-Entropy Loss -> Combined KD Loss -> Update Student Model
