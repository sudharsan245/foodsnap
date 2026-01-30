# Food-Specific AI Detection - Fine-Tuning Guide

This guide covers the complete process of adapting the SMOGY model for food-specific AI image detection through domain adaptation and fine-tuning.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Preparation](#dataset-preparation)
4. [Phase 1: Model Inspection](#phase-1-model-inspection)
5. [Phase 2: Dataset Integration](#phase-2-dataset-integration)
6. [Phase 3: Fine-Tuning](#phase-3-fine-tuning)
7. [Phase 4: Hard Negatives](#phase-4-hard-negatives)
8. [Phase 5: Evaluation](#phase-5-evaluation)
9. [Integration](#integration)
10. [Troubleshooting](#troubleshooting)

---

## Overview

**Goal**: Adapt SMOGY (general AI-vs-Real detector) → Food-specific AI detector

**Strategy**:
- ✅ Freeze 80-90% of model (preserve AI detection knowledge)
- ✅ Train final layers on food-specific data
- ✅ 3-class classification: Real Clean, Real Contaminated, AI-Generated
- ✅ Heavy augmentation for robustness
- ✅ Low false positive rate (<5%)

**Key Principle**: *"Learn the absence of physical cooking chaos and camera sensor randomness"*

---

## Prerequisites

### System Requirements

```bash
# GPU recommended (but CPU works)
- CUDA-capable GPU with 8GB+ VRAM (recommended)
- 16GB+ RAM
- 50GB+ disk space
```

### Install Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements_finetuning.txt
```

**requirements_finetuning.txt**:
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.13.0
tqdm>=4.65.0
```

---

## Dataset Preparation

### Directory Structure

Create your dataset with the following structure:

```
food_ai_dataset/
├── real_clean/
│   ├── normal_food_001.jpg
│   ├── normal_food_002.jpg
│   └── ...
├── real_contaminated/
│   ├── hair_in_food_001.jpg
│   ├── insect_001.jpg
│   ├── mold_001.jpg
│   ├── plastic_piece_001.jpg
│   ├── metal_fragment_001.jpg
│   ├── burnt_food_001.jpg
│   └── ...
└── ai_generated/
    ├── midjourney_food_001.jpg
    ├── dalle_food_001.jpg
    ├── stable_diffusion_001.jpg
    ├── ai_contaminated_001.jpg  # AI with fake contamination
    └── ...
```

### Dataset Requirements

**Class 0: Real Clean Food**
- Normal food delivery photos
- Various cuisines
- Different lighting conditions
- Multiple camera types

**Class 1: Real Contaminated Food**
- **Must include visible contamination:**
  - Hair (human/animal)
  - Insects/bugs
  - Mold/fungus
  - Plastic pieces
  - Metal fragments
  - Burnt/charred food
  - Foreign objects

**Class 2: AI-Generated Food**
- Images from various AI generators:
  - Midjourney
  - DALL-E 3
  - Stable Diffusion
  - Other generative models
- Both clean and fake-contaminated AI food
- Various styles (photorealistic, artistic)

### Recommended Dataset Size

| Split | Real Clean | Real Contaminated | AI-Generated | Total |
|-------|-----------|-------------------|--------------|-------|
| Train | 1,000+ | 1,000+ | 1,000+ | 3,000+ |
| Val | 200+ | 200+ | 200+ | 600+ |
| Test | 200+ | 200+ | 200+ | 600+ |

**Minimum**: 300 images per class (900 total)  
**Recommended**: 1,000+ images per class (3,000+ total)

---

## Phase 1: Model Inspection

### Inspect Model Architecture

```bash
# View complete model architecture
python inspect_model.py --inspect

# View layer freezing plan
python inspect_model.py --freeze

# Apply freezing and save configuration
python inspect_model.py --apply

# Run all steps
python inspect_model.py --all
```

### Expected Output

```
📊 Parameter Statistics:
   Total:        28,000,000 params
   Trainable:     4,200,000 params (15.0%)
   Frozen:       23,800,000 params (85.0%)

🔒 LAYERS TO FREEZE:
   Categories: patch_embed, stage_1, stage_2, stage_3
   
🔓 LAYERS TO TRAIN:
   Categories: stage_4, classification_head
```

### Output Files

- `freezing_config.json` - Layer freezing configuration for training

---

## Phase 2: Dataset Integration

### Test Dataset Loading

```python
from dataset import create_dataloaders

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    data_dir='./food_ai_dataset',
    batch_size=16,
    balance_classes=True
)

# Check one batch
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    break
```

### Augmentation Pipeline

The dataset automatically applies:
- ✅ JPEG compression (quality 60-95)
- ✅ Gaussian blur
- ✅ Random crop/zoom
- ✅ Low-light simulation
- ✅ Color distortion
- ✅ Rotation (±15°)
- ✅ Horizontal flip

---

## Phase 3: Fine-Tuning

### Basic Training

```bash
python finetune.py \
    --data_dir ./food_ai_dataset \
    --epochs 15 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --output_dir ./checkpoints
```

### Advanced Training Options

```bash
python finetune.py \
    --data_dir ./food_ai_dataset \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 5e-6 \
    --output_dir ./checkpoints \
    --freeze_config ./freezing_config.json \
    --early_stopping 5
```

### Resume Training

```bash
python finetune.py \
    --data_dir ./food_ai_dataset \
    --resume ./checkpoints/checkpoint_epoch_10.pth
```

### Monitor Training

```bash
# In a separate terminal
tensorboard --logdir ./checkpoints/logs
```

Open http://localhost:6006 to view:
- Training/validation loss curves
- Per-class accuracy
- Learning rate schedule

### Expected Training Time

| Hardware | Batch Size | Time per Epoch | Total (15 epochs) |
|----------|-----------|----------------|-------------------|
| RTX 3090 | 32 | ~5 min | ~75 min |
| RTX 3060 | 16 | ~10 min | ~150 min |
| CPU | 8 | ~60 min | ~15 hours |

### Output Files

```
checkpoints/
├── best_model.pth              # Best validation accuracy
├── checkpoint_epoch_1.pth      # Epoch checkpoints
├── checkpoint_epoch_2.pth
├── ...
└── logs/
    └── 20260130_120000/        # TensorBoard logs
```

---

## Phase 4: Hard Negatives

### Hard Negative Examples to Include

**1. Real Food with Heavy Filters**
- Instagram-filtered food photos
- Heavily edited images
- HDR/beauty mode photos

**2. Screenshots**
- Screenshots of food photos
- WhatsApp forwarded images
- Social media screenshots

**3. Re-photographed Prints**
- Photos of printed food images
- Screen captures
- Photos of photos

**4. Post-Processed AI Images**
- AI images with added noise
- AI images with blur/compression
- AI images with filters

**5. Extreme Close-ups**
- Macro shots of food
- Cropped regions
- Limited context images

### Adding Hard Negatives

```
food_ai_dataset/
├── real_clean/
│   ├── filtered_food_001.jpg      # Instagram filters
│   ├── screenshot_001.jpg         # Screenshots
│   ├── reprinted_001.jpg          # Re-photographed
│   └── closeup_001.jpg            # Extreme close-ups
└── ai_generated/
    ├── ai_with_noise_001.jpg      # Post-processed AI
    ├── ai_blurred_001.jpg
    └── ai_compressed_001.jpg
```

**Retrain** after adding hard negatives:

```bash
python finetune.py \
    --data_dir ./food_ai_dataset \
    --epochs 10 \
    --resume ./checkpoints/best_model.pth
```

---

## Phase 5: Evaluation

### Run Evaluation

```bash
python evaluate.py \
    --model_path ./checkpoints/best_model.pth \
    --data_dir ./test_data \
    --output_dir ./evaluation_results
```

### Evaluation Metrics

The script calculates:

**Critical Metrics:**
- ✅ False Positive Rate (Real → AI): **Target <5%**
- ✅ AI Detection Recall: **Target >80%**
- ✅ Per-class Precision/Recall/F1

**Confidence Analysis:**
- Mean confidence per class
- Confidence distribution
- Threshold calibration

### Output Files

```
evaluation_results/
├── evaluation_metrics.json       # All metrics
├── confusion_matrix.png          # Confusion matrix heatmap
├── confidence_distribution.png   # Confidence histograms
└── confidence_boxplot.png        # Box plots
```

### Interpreting Results

**Good Performance:**
```
False Positive Rate: 3.2%  ✅
AI Detection Recall: 87.5% ✅
Overall Accuracy: 91.2%    ✅
```

**Needs Improvement:**
```
False Positive Rate: 12.5% ❌ (Too many false positives)
AI Detection Recall: 65.0% ⚠️ (Missing AI images)
```

### Threshold Calibration

Based on evaluation results, adjust thresholds in `config.py`:

```python
# Conservative (minimize false positives)
REJECT_THRESHOLD = 0.90
MANUAL_REVIEW_THRESHOLD = 0.70

# Balanced (default)
REJECT_THRESHOLD = 0.85
MANUAL_REVIEW_THRESHOLD = 0.60

# Aggressive (maximize AI detection)
REJECT_THRESHOLD = 0.80
MANUAL_REVIEW_THRESHOLD = 0.50
```

---

## Integration

### Update Model Path

After training, update `config.py`:

```python
# Before (pretrained SMOGY)
MODEL_ID = "Smogy/SMOGY-Ai-images-detector"

# After (fine-tuned model)
MODEL_ID = "./checkpoints/best_model.pth"
```

### Modify Detector to Load Fine-Tuned Model

Update `detector.py` to handle local checkpoint:

```python
# In FoodImageDetector.__init__
if Path(model_id).exists():
    # Load from local checkpoint
    checkpoint = torch.load(model_id, map_location=self.device)
    self.model = AutoModelForImageClassification.from_pretrained(
        "Smogy/SMOGY-Ai-images-detector"
    )
    # Modify for 3 classes
    if self.model.config.num_labels != 3:
        import torch.nn as nn
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, 3)
        self.model.config.num_labels = 3
    
    self.model.load_state_dict(checkpoint['model_state_dict'])
else:
    # Load from Hugging Face
    self.model = AutoModelForImageClassification.from_pretrained(model_id)
```

### Test Integration

```bash
# Test with Flask app
python app.py
```

Upload test images and verify:
- ✅ Model loads successfully
- ✅ Predictions work
- ✅ Thresholds are correct
- ✅ Response format unchanged

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
```bash
# Reduce batch size
python finetune.py --batch_size 8

# Use gradient accumulation (simulate larger batch)
# Modify finetune.py to accumulate gradients
```

### Issue: Model Not Learning

**Symptoms:**
- Validation accuracy stuck
- Loss not decreasing

**Solutions:**
1. Check dataset balance
2. Increase learning rate: `--learning_rate 5e-5`
3. Unfreeze more layers
4. Add more training data

### Issue: High False Positive Rate

**Solutions:**
1. Add more hard negative examples
2. Increase REJECT_THRESHOLD
3. Add more real food examples with filters/effects
4. Retrain with class weights favoring real food

### Issue: Low AI Detection Recall

**Solutions:**
1. Add more diverse AI-generated examples
2. Include AI images from newer generators
3. Decrease REJECT_THRESHOLD
4. Add adversarial AI examples (post-processed)

### Issue: Slow Training on CPU

**Solutions:**
1. Reduce batch size: `--batch_size 4`
2. Reduce image size in dataset.py: `image_size=192`
3. Use fewer workers: `num_workers=2`
4. Consider cloud GPU (Google Colab, AWS, etc.)

---

## Best Practices

### 1. Data Quality
- ✅ Diverse real food images (various cuisines, lighting, cameras)
- ✅ Clear contamination in Class 1 (visible to human eye)
- ✅ Diverse AI generators in Class 2
- ✅ Balanced class distribution

### 2. Training
- ✅ Start with low learning rate (1e-5)
- ✅ Use early stopping (patience=5)
- ✅ Monitor validation metrics closely
- ✅ Save checkpoints frequently

### 3. Evaluation
- ✅ Test on completely separate data
- ✅ Simulate real-world conditions (compression, screenshots)
- ✅ Prioritize low false positive rate
- ✅ Document failure cases

### 4. Deployment
- ✅ Conservative thresholds initially
- ✅ Monitor real-world performance
- ✅ Collect edge cases for retraining
- ✅ Regular model updates

---

## Next Steps After Fine-Tuning

1. **Deploy to Production**
   - Update `config.py` with fine-tuned model path
   - Test thoroughly with real complaint images
   - Set up monitoring and logging

2. **Continuous Improvement**
   - Collect misclassified examples
   - Retrain periodically with new data
   - Track model performance metrics

3. **A/B Testing**
   - Compare fine-tuned vs. original model
   - Measure business impact (fraud reduction)
   - Adjust thresholds based on results

---

## Summary

**Complete Workflow:**

```bash
# 1. Prepare dataset
mkdir -p food_ai_dataset/{real_clean,real_contaminated,ai_generated}
# ... add images ...

# 2. Inspect model
python inspect_model.py --all

# 3. Train model
python finetune.py --data_dir ./food_ai_dataset --epochs 15

# 4. Evaluate model
python evaluate.py \
    --model_path ./checkpoints/best_model.pth \
    --data_dir ./test_data

# 5. Update config and deploy
# Edit config.py: MODEL_ID = "./checkpoints/best_model.pth"
python app.py
```

**Key Success Metrics:**
- ✅ False Positive Rate < 5%
- ✅ AI Detection Recall > 80%
- ✅ Overall Accuracy > 85%

---

**Questions or Issues?**
- Review the DOMAIN_ADAPTATION_PLAN.md for detailed strategy
- Check TensorBoard logs for training insights
- Examine evaluation_results/ for performance analysis

**Remember**: *"The model learns the absence of physical cooking chaos and camera sensor randomness"*
