# Food-Specific AI Image Detector - Domain Adaptation Plan

## Executive Summary

This document outlines the complete strategy for adapting the SMOGY AI-vs-Real image detection model into a **food-specific AI image detector** through targeted fine-tuning and domain adaptation.

**Core Principle**: Preserve the model's learned general AI-vs-Real texture knowledge while adapting it to food-specific scenarios, including real contamination cases.

---

## PHASE 1: Model Inspection & Freezing Strategy

### 1.1 SMOGY Architecture Overview

The SMOGY model is based on **Swin Transformer** architecture:

```
Input Image (224x224)
    ↓
Patch Embedding Layer (converts image to patches)
    ↓
Swin Transformer Blocks (4 stages)
    ├── Stage 1: Early features (edges, textures)
    ├── Stage 2: Mid-level features (patterns)
    ├── Stage 3: High-level features (objects)
    └── Stage 4: Abstract features (semantic understanding)
    ↓
Classification Head (Linear layer → 2 classes)
```

### 1.2 Layer Freezing Strategy

**Freeze (80-90% of parameters):**
- ✅ Patch embedding layers (preserve low-level feature extraction)
- ✅ Stage 1 & 2 transformer blocks (preserve texture/pattern detection)
- ✅ Stage 3 transformer blocks (preserve object understanding)

**Keep Trainable (10-20% of parameters):**
- 🔓 Stage 4 transformer blocks (adapt to food-specific features)
- 🔓 Classification head (learn food-specific decision boundaries)

**Rationale:**
- Early layers learn universal features (edges, textures, compression artifacts)
- These are critical for detecting AI generation artifacts
- Only final layers need adaptation for food domain specifics
- Prevents catastrophic forgetting of AI detection capabilities

### 1.3 Expected Trainable Parameters

```
Total Parameters: ~28M (Swin-Tiny)
Frozen Parameters: ~24M (85%)
Trainable Parameters: ~4M (15%)
```

---

## PHASE 2: Food-Specific Dataset Integration

### 2.1 Dataset Classes

**3-Class Classification:**

| Class | Label | Description | Examples |
|-------|-------|-------------|----------|
| 0 | Real Food (Clean) | Genuine food photos without contamination | Normal delivery photos |
| 1 | Real Food (Contaminated) | Genuine photos with visible contamination | Hair, insects, mold, plastic, metal, burnt food |
| 2 | AI-Generated Food | AI-created food images (clean or fake-contaminated) | Midjourney, DALL-E, Stable Diffusion outputs |

### 2.2 Dataset Requirements

**Real Contaminated Food (Class 1):**
- ✅ Visible hair in food
- ✅ Insects/bugs
- ✅ Mold/fungus
- ✅ Plastic pieces
- ✅ Metal fragments
- ✅ Burnt/charred food
- ✅ Foreign objects

**AI-Generated Food (Class 2):**
- ✅ Clean AI food images
- ✅ AI food with fake contamination
- ✅ Various AI generators (Midjourney, DALL-E, Stable Diffusion)
- ✅ Different styles (photorealistic, artistic)

### 2.3 Data Augmentation Pipeline

**Heavy Real-World Augmentations:**

```python
Augmentations:
├── JPEG Compression (quality 60-95)
├── Gaussian Blur (sigma 0.5-2.0)
├── Random Crop/Zoom (0.8-1.0 scale)
├── Low-light Simulation (brightness 0.5-1.5)
├── Color Distortion (hue, saturation, contrast)
├── Random Rotation (±15°)
├── Perspective Transform
└── Noise Injection (Gaussian, salt-pepper)
```

**Purpose**: Simulate real-world complaint image conditions (WhatsApp compression, poor lighting, phone camera quality)

### 2.4 Class Balancing

**Strategy:**
- Equal samples per class during training
- Oversample minority classes if needed
- Weighted loss function to handle imbalance

**Target Distribution:**
- Class 0 (Real Clean): 33%
- Class 1 (Real Contaminated): 33%
- Class 2 (AI-Generated): 34%

---

## PHASE 3: Fine-Tuning Procedure

### 3.1 Training Configuration

```yaml
Training Setup:
  optimizer: AdamW
  learning_rate: 1e-5  # Low LR for fine-tuning
  weight_decay: 0.01
  batch_size: 16
  epochs: 10-15
  loss: CrossEntropyLoss (with class weights)
  scheduler: CosineAnnealingLR
  warmup_steps: 500
```

### 3.2 Training Strategy

**Phase 3A: Initial Fine-Tuning (3-class)**
1. Train on balanced 3-class dataset
2. Monitor validation accuracy per class
3. Early stopping on validation loss plateau
4. Save best checkpoint

**Phase 3B: Optional Binary Conversion**
- Merge Class 0 & 1 → REAL
- Class 2 → AI-GENERATED
- Retrain classification head only (2-3 epochs)

### 3.3 Evaluation Metrics

**Primary Metrics:**
- Precision (minimize false positives on real food)
- Recall (catch AI-generated images)
- F1-Score per class
- Confusion Matrix

**Critical Constraint:**
- **False Positive Rate on Real Food < 5%**
- Better to miss AI images than wrongly reject real complaints

### 3.4 Overfitting Prevention

- ✅ Heavy augmentation
- ✅ Dropout in classification head
- ✅ Early stopping
- ✅ Validation on held-out set
- ✅ Test on completely different data sources

---

## PHASE 4: Food-Specific Negative Knowledge

### 4.1 Hard Negative Examples

**Include in training:**

| Hard Negative Type | Why It's Hard | Solution |
|-------------------|---------------|----------|
| Real food with heavy Instagram filters | May look "too perfect" | Include in Class 0/1 with augmentation |
| Screenshots of food photos | Compression artifacts similar to AI | Explicit screenshot examples |
| Printed photos re-captured | Texture changes confuse model | Include re-photographed prints |
| AI food post-processed to look real | Noise/blur added to AI images | Include adversarial AI examples |
| Extreme close-ups | Limited context | Crop augmentation on real images |

### 4.2 Adversarial Training

**Strategy:**
1. Generate AI food images
2. Post-process with noise, blur, compression
3. Include as Class 2 (AI) in training
4. Forces model to detect subtle AI artifacts

### 4.3 Expected Robustness Gains

- ✅ Reduced false positives on filtered real photos
- ✅ Better handling of low-quality images
- ✅ Resistance to simple post-processing tricks
- ✅ Improved generalization to new AI generators

---

## PHASE 5: Real-World Validation

### 5.1 Test Conditions

**Simulate Real Complaint Scenarios:**

| Test Scenario | Image Source | Expected Challenge |
|--------------|--------------|-------------------|
| WhatsApp screenshots | Compressed, re-encoded | Multiple compression layers |
| Zomato/Swiggy uploads | Mobile app compression | Platform-specific artifacts |
| Low-resolution images | <500px | Limited detail |
| Cropped food regions | Only food, no context | Missing environmental cues |
| Night/low-light photos | Poor lighting | High noise, low contrast |

### 5.2 Evaluation Metrics

**Critical Metrics:**

```python
Metrics:
├── Precision (Real Food): > 95%
├── Recall (AI Food): > 80%
├── False Positive Rate: < 5%
├── Confidence Score Distribution
│   ├── Real Food: Mean confidence > 0.7
│   └── AI Food: Mean confidence > 0.8
└── Threshold Analysis
    ├── P(AI) < 0.60 → Accept
    ├── 0.60 ≤ P(AI) < 0.80 → Manual Review
    └── P(AI) ≥ 0.80 → Likely AI
```

### 5.3 Failure Case Analysis

**Document and analyze:**
- False positives (real food flagged as AI)
- False negatives (AI food accepted as real)
- Edge cases requiring manual review
- Systematic biases (e.g., certain cuisines)

### 5.4 Threshold Calibration

**Adjust thresholds based on business requirements:**

| Threshold Set | Use Case | FPR | Recall |
|--------------|----------|-----|--------|
| Conservative | Customer-first approach | <2% | ~70% |
| Balanced | Default setting | ~5% | ~85% |
| Aggressive | High fraud environment | ~10% | ~95% |

---

## FINAL CONSTRAINTS & INTEGRATION

### Constraints

✅ **No EXIF/Metadata**: Detection is purely pixel-based  
✅ **No Redesign**: Integrates with existing Flask/frontend  
✅ **Seamless Integration**: Drop-in replacement for current model  
✅ **Backward Compatible**: Same API interface  

### Integration Points

```python
# detector.py - No changes to API
detector = FoodImageDetector()  # Now loads fine-tuned model
result = detector.predict("image.jpg")  # Same interface

# Only change: MODEL_ID in config.py
MODEL_ID = "path/to/fine-tuned-food-model"
```

### Deployment Strategy

1. Train and validate fine-tuned model
2. Save model to local path or Hugging Face Hub
3. Update `MODEL_ID` in `config.py`
4. Test with existing Flask app
5. Deploy with same infrastructure

---

## FINAL EXPLANATION

**"The model is a food-domain–fine-tuned Swin Transformer that detects AI-generated food images by learning the absence of physical cooking chaos and camera sensor randomness."**

### What This Means:

**Real Food Photos Contain:**
- ✅ Camera sensor noise patterns
- ✅ Natural lighting inconsistencies
- ✅ Physical texture randomness (steam, grease, crumbs)
- ✅ Environmental context (plates, tables, hands)
- ✅ Compression artifacts from real camera → upload pipeline

**AI-Generated Food Images Lack:**
- ❌ True sensor noise (synthetic noise is different)
- ❌ Physical cooking chaos (too perfect, unrealistic lighting)
- ❌ Authentic texture randomness (patterns are learned, not physical)
- ❌ Real-world imperfections (scratches on plates, fingerprints)
- ❌ Natural compression artifacts (AI → save → upload has different signature)

**The Fine-Tuned Model Learns:**
- 🎯 Food-specific AI generation artifacts
- 🎯 Difference between real contamination and AI-faked contamination
- 🎯 Robustness to real-world image degradation
- 🎯 Conservative decision boundaries (favor customers)

---

## Next Steps

1. **Implement model inspection script** (`inspect_model.py`)
2. **Create dataset loader** (`dataset.py`)
3. **Implement fine-tuning script** (`finetune.py`)
4. **Build evaluation pipeline** (`evaluate.py`)
5. **Test integration** with existing Flask app

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-30  
**Status**: Ready for Implementation
