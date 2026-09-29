# Efficient Satellite Image Classification via Knowledge Distillation

A PyTorch-based investigation of knowledge distillation techniques for compressing a high-performance Multi-Scale Dynamic Fusion (MSDF) Swin Transformer into lightweight ResNet-18 student models for remote-sensing image classification.

The project evaluates whether smaller convolutional models can retain the classification performance of a substantially larger transformer while reducing model size and training cost.

## Key Results

The strongest student model used layer-wise knowledge distillation and achieved:

| Model | EuroSAT Accuracy | Parameters |
| --- | ---: | ---: |
| MSDF Swin Transformer Teacher | 98.50% | 28.975M |
| Layer-Wise KD ResNet-18 | **98.04%** | **11.71M** |
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

The strongest-performing student used layer-wise distillation and combined three complementary learning signals:

- **Cross-Entropy Loss** — learns from ground-truth class labels
- **KL-Divergence Loss** — transfers the teacher's softened class probability distribution
- **Feature MSE Loss** — aligns intermediate student feature representations with teacher feature maps

The layer-wise loss combines both final model predictions and intermediate representations, allowing the student to learn more information than response-based distillation alone.

```md
## Knowledge Distillation Pipeline

```text
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
            |                       |
     Teacher Feature Maps     Student Feature Maps
            |                       |
            +------ MSE Loss -------+
                                    |
Ground-Truth Labels ----------------+
                                    |
                         Cross-Entropy Loss
                                    |
                                    v
                           Combined KD Loss
                                    |
                                    v
                         Update Student Model
```


The student is optimized using a composite objective that combines ground-truth classification loss, teacher-student logit distillation, and intermediate feature-map alignment.

## Results Interpretation

The layer-wise student achieved 98.04% accuracy on EuroSAT compared with 98.50% for the MSDF Swin teacher, while reducing the model from approximately 28.98M to 11.71M parameters.

This suggests that intermediate feature alignment can preserve most of the teacher model's predictive performance while substantially reducing model size.

## Models

### `teacher_model.py`
Implements the MSDF Swin Transformer teacher using a Swin-Tiny backbone, multi-scale feature fusion, and attention/feature extraction.

### `Baseline_ResNet.py`
Implements the standard ResNet-18 baseline used for comparison.

### `resnet_LayerWise_distilled.py`
Implements the layer-wise knowledge-distilled ResNet-18 using:
- Cross-entropy loss
- Temperature-scaled KL-divergence
- Intermediate feature-map MSE loss

### `MSDFResNet_LayerWise_Distilled.py`
Extends the ResNet-18 student with an MSDF module and additional distillation across fused features and gating outputs.


