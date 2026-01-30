# AI-Generated Image Detection for Food Delivery Fraud Prevention

A production-ready Python system to detect AI-generated food images using the **SMOGY** model from Hugging Face.

## 🎯 Purpose

In food delivery applications, users sometimes submit false claims with AI-generated images (e.g., food with fake contaminants) to get fraudulent refunds. This system automatically detects whether a submitted image is AI-generated or real.

## 🔬 Food-Specific Fine-Tuning (NEW!)

The base SMOGY model can be **fine-tuned for food-specific AI detection** to improve accuracy on food images and handle real contamination cases.

**Key Features:**
- ✅ **3-Class Classification**: Real Clean, Real Contaminated, AI-Generated
- ✅ **Domain Adaptation**: Preserves AI detection knowledge while learning food-specific features
- ✅ **Robust to Real-World Conditions**: Heavy augmentation for WhatsApp compression, screenshots, low-light
- ✅ **Low False Positives**: Prioritizes not rejecting legitimate customer complaints (<5% FPR target)

**Quick Start:**
```bash
# Interactive setup
python quickstart_finetuning.py

# Or manual workflow
python inspect_model.py --all          # Phase 1: Inspect model
python finetune.py --data_dir ./data   # Phase 3: Train
python evaluate.py --model_path ./checkpoints/best_model.pth --data_dir ./test_data
```

**📚 See [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) for complete instructions**

**📋 See [DOMAIN_ADAPTATION_PLAN.md](DOMAIN_ADAPTATION_PLAN.md) for technical strategy**

## 📦 Installation

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
pillow>=10.0.0
```

### Install Dependencies

```bash
pip install torch torchvision transformers pillow
```

## 🚀 Quick Start

### Single Image Analysis

```python
from detector import FoodImageDetector

# Initialize detector (auto-selects GPU if available)
detector = FoodImageDetector()

# Analyze an image
result = detector.predict("suspicious_food.jpg")

print(f"AI Probability: {result.ai_probability:.2%}")
print(f"Decision: {result.decision}")
```

### Command Line Usage

```bash
# Test model loading
python main.py --test

# Analyze single image
python main.py --image food_photo.jpg

# Analyze multiple images
python main.py --images img1.jpg img2.jpg img3.jpg

# Analyze all images in a directory
python main.py --directory ./claim_images/

# Get JSON output
python main.py --image food.jpg --json

# Force CPU mode
python main.py --image food.jpg --device cpu
```

## 📊 Decision Logic

The system uses confidence thresholds to make decisions:

| AI Confidence | Decision | Action |
|--------------|----------|--------|
| ≥ 85% | 🔴 **REJECT** | Block refund, flag account |
| 60% - 85% | 🟡 **MANUAL REVIEW** | Queue for human verification |
| < 60% | 🟢 **ACCEPT** | Process refund normally |

### Customizing Thresholds

```python
from detector import FoodImageDetector
from config import DecisionThresholds

# More conservative thresholds (fewer rejections)
custom_thresholds = DecisionThresholds(
    REJECT_THRESHOLD=0.95,       # Only reject if 95%+ confident
    MANUAL_REVIEW_THRESHOLD=0.70  # Review between 70-95%
)

detector = FoodImageDetector(thresholds=custom_thresholds)
```

## 🔍 Understanding Results

```python
result = detector.predict("image.jpg")

# Access all properties
print(result.image_path)        # Path to analyzed image
print(result.ai_probability)    # 0.0 to 1.0
print(result.real_probability)  # 0.0 to 1.0
print(result.probabilities)     # {"artificial": 0.xx, "real": 0.xx}
print(result.decision)          # ClaimDecision.ACCEPT/REJECT/MANUAL_REVIEW
print(result.decision.emoji)    # 🟢/🔴/🟡

# Convert to dictionary (for APIs)
data = result.to_dict()
```

## 📦 Batch Processing

For processing multiple images efficiently:

```python
# Process images in batches of 8
results = detector.predict_batch(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg", ...],
    batch_size=8
)

for result in results:
    if result.decision == ClaimDecision.REJECT:
        handle_fraudulent_claim(result)
```

## ⚠️ Important Limitations

### Technical Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Trained on general images** | May not perfectly generalize to food photos | Use conservative thresholds favoring customers |
| **JPG compression** | Heavy compression can confuse the model | Request high-quality uploads |
| **Screenshots** | Real photos may appear synthetic after screenshot | Always provide manual review path |
| **New AI generators** | May not detect images from latest models | Regular model updates required |

### Ethical Guidelines

1. **Never auto-reject without recourse** - Always provide an appeal mechanism
2. **Transparency** - Inform users that images may be analyzed for authenticity
3. **Human oversight** - All rejections should be reviewable by humans
4. **Privacy** - Process images in-memory, don't store for detection purposes
5. **Bias monitoring** - Regularly audit for demographic or content biases

## ⚖️ License Note

> ⚠️ The SMOGY model is licensed for **non-commercial use only** due to its training data sources. For production deployment in commercial applications, consult your legal team or consider alternative models.

## 📁 Project Structure

```
food image detection/
├── config.py      # Configuration, thresholds, enums
├── detector.py    # Main FoodImageDetector class
├── main.py        # CLI entry point
└── README.md      # This file
```

## 🛠️ API Reference

### FoodImageDetector

```python
class FoodImageDetector:
    def __init__(
        self,
        model_id: str = "Smogy/SMOGY-Ai-images-detector",
        thresholds: DecisionThresholds = DEFAULT_THRESHOLDS,
        device: str = None  # "cuda", "cpu", or None for auto
    )
    
    def predict(self, image_path: str) -> DetectionResult
    def predict_batch(self, image_paths: List[str], batch_size: int = 8) -> List[DetectionResult]
    def get_model_info(self) -> dict
```

### DetectionResult

```python
@dataclass
class DetectionResult:
    image_path: str
    probabilities: dict
    ai_probability: float
    real_probability: float
    decision: ClaimDecision
    ai_label: str
    real_label: str
    
    def to_dict(self) -> dict
```

### ClaimDecision

```python
class ClaimDecision(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    MANUAL_REVIEW = "manual_review"
```
