"""
AI-Generated Image Detector for Food Delivery Fraud Prevention.

This module provides a production-ready detector class for identifying
AI-generated food images to prevent fraudulent refund claims.

Key Features:
    - Automatic CPU/GPU device selection
    - Single and batch image inference
    - Configurable decision thresholds
    - Detailed detection results with probabilities

Usage:
    from detector import FoodImageDetector
    
    detector = FoodImageDetector()
    result = detector.predict("path/to/food_image.jpg")
    print(result)
"""

import torch
from PIL import Image
from pathlib import Path
from typing import List, Union, Optional
from dataclasses import dataclass, field
from transformers import AutoModelForImageClassification, AutoImageProcessor

from config import MODEL_ID, ClaimDecision, DecisionThresholds, DEFAULT_THRESHOLDS


# =============================================================================
# Detection Result Data Class
# =============================================================================

@dataclass
class DetectionResult:
    """
    Result of AI image detection inference.
    
    Attributes:
        image_path: Path to the analyzed image
        probabilities: Dictionary with class probabilities (keys from model config)
        ai_probability: Probability that image is AI-generated (0.0 to 1.0)
        real_probability: Probability that image is real (0.0 to 1.0)
        decision: Claim decision based on thresholds (ACCEPT/REJECT/MANUAL_REVIEW)
        ai_label: The actual label name for AI class from model config
        real_label: The actual label name for Real class from model config
    """
    image_path: str
    probabilities: dict
    ai_probability: float
    real_probability: float
    decision: ClaimDecision
    ai_label: str = "artificial"
    real_label: str = "real"
    
    def __str__(self) -> str:
        return (
            f"DetectionResult(\n"
            f"  image: {Path(self.image_path).name}\n"
            f"  {self.ai_label}: {self.ai_probability:.2%}\n"
            f"  {self.real_label}: {self.real_probability:.2%}\n"
            f"  decision: {self.decision.emoji} {self.decision.description}\n"
            f")"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "image_path": self.image_path,
            "probabilities": self.probabilities,
            "ai_probability": self.ai_probability,
            "real_probability": self.real_probability,
            "decision": self.decision.value,
            "ai_label": self.ai_label,
            "real_label": self.real_label
        }


# =============================================================================
# Main Detector Class
# =============================================================================

class FoodImageDetector:
    """
    Detector for AI-generated food images using the SMOGY model.
    
    This class provides methods to detect whether food images submitted
    for refund claims are AI-generated or real photographs.
    
    Attributes:
        model: The loaded Hugging Face image classification model
        processor: Image processor for preprocessing
        device: Torch device (cuda or cpu)
        thresholds: Decision thresholds for claim logic
        label_mapping: Mapping of model labels to semantic meaning
    
    Example:
        >>> detector = FoodImageDetector()
        >>> result = detector.predict("suspicious_food.jpg")
        >>> if result.decision == ClaimDecision.REJECT:
        ...     print("Fraudulent claim detected!")
    """
    
    def __init__(
        self,
        model_id: str = MODEL_ID,
        thresholds: DecisionThresholds = DEFAULT_THRESHOLDS,
        device: Optional[str] = None
    ):
        """
        Initialize the detector with model and configuration.
        
        Args:
            model_id: Hugging Face model identifier
            thresholds: Decision thresholds for claim logic
            device: Force specific device ('cuda' or 'cpu'), auto-detect if None
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Loading model on device: {self.device}")
        
        # Load model and processor from Hugging Face
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Store thresholds
        self.thresholds = thresholds
        
        # Extract label mapping from model config
        self._setup_label_mapping()
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Labels: {self.model.config.id2label}")
    
    def _setup_label_mapping(self) -> None:
        """
        Setup label mapping from model config.
        
        The model's id2label config tells us which index corresponds to which class.
        We need to identify which label represents "AI/artificial" and which is "real".
        """
        id2label = self.model.config.id2label
        
        # Common variations of AI/fake labels
        ai_keywords = ["artificial", "ai", "fake", "generated", "synthetic"]
        real_keywords = ["real", "human", "authentic", "genuine", "natural"]
        
        self.ai_label = None
        self.real_label = None
        self.ai_index = None
        self.real_index = None
        
        for idx, label in id2label.items():
            label_lower = label.lower()
            
            # Check for AI label
            if any(kw in label_lower for kw in ai_keywords):
                self.ai_label = label
                self.ai_index = int(idx)
            
            # Check for Real label
            if any(kw in label_lower for kw in real_keywords):
                self.real_label = label
                self.real_index = int(idx)
        
        # Fallback: assume index 0 is AI, index 1 is Real (common convention)
        if self.ai_label is None or self.real_label is None:
            print(f"⚠️ Could not auto-detect labels, using index-based fallback")
            print(f"   Model labels: {id2label}")
            self.ai_index = 0
            self.real_index = 1
            self.ai_label = id2label.get(0, id2label.get("0", "class_0"))
            self.real_label = id2label.get(1, id2label.get("1", "class_1"))
        
        print(f"🏷️ Label mapping: AI='{self.ai_label}' (idx {self.ai_index}), Real='{self.real_label}' (idx {self.real_index})")
    
    def _load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load and validate an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image in RGB format
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If file is not a valid image
        """
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        try:
            image = Image.open(path).convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image {path}: {e}")
    
    def _make_decision(self, ai_probability: float) -> ClaimDecision:
        """
        Make claim decision based on AI probability and thresholds.
        
        Args:
            ai_probability: Probability that image is AI-generated (0.0 to 1.0)
            
        Returns:
            ClaimDecision enum value
        """
        if ai_probability >= self.thresholds.REJECT_THRESHOLD:
            return ClaimDecision.REJECT
        elif ai_probability >= self.thresholds.MANUAL_REVIEW_THRESHOLD:
            return ClaimDecision.MANUAL_REVIEW
        else:
            return ClaimDecision.ACCEPT
    
    def predict(self, image_path: Union[str, Path]) -> DetectionResult:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to the image file to analyze
            
        Returns:
            DetectionResult with probabilities and decision
            
        Example:
            >>> result = detector.predict("food_photo.jpg")
            >>> print(f"AI probability: {result.ai_probability:.2%}")
            >>> print(f"Decision: {result.decision}")
        """
        image_path = str(image_path)
        image = self._load_image(image_path)
        
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]
        
        # Extract probabilities
        probs = probabilities.cpu().numpy()
        ai_prob = float(probs[self.ai_index])
        real_prob = float(probs[self.real_index])
        
        # Build probability dictionary
        prob_dict = {
            self.ai_label: ai_prob,
            self.real_label: real_prob
        }
        
        # Make decision
        decision = self._make_decision(ai_prob)
        
        return DetectionResult(
            image_path=image_path,
            probabilities=prob_dict,
            ai_probability=ai_prob,
            real_probability=real_prob,
            decision=decision,
            ai_label=self.ai_label,
            real_label=self.real_label
        )
    
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 8
    ) -> List[DetectionResult]:
        """
        Run inference on multiple images efficiently.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Number of images to process at once
            
        Returns:
            List of DetectionResult objects
            
        Example:
            >>> results = detector.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
            >>> for r in results:
            ...     print(f"{r.image_path}: {r.decision}")
        """
        results = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_paths = []
            
            # Load images, skip invalid ones
            for path in batch_paths:
                try:
                    image = self._load_image(path)
                    batch_images.append(image)
                    valid_paths.append(str(path))
                except (FileNotFoundError, ValueError) as e:
                    print(f"⚠️ Skipping {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Preprocess batch
            inputs = self.processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Process each result
            probs_np = probabilities.cpu().numpy()
            for j, path in enumerate(valid_paths):
                ai_prob = float(probs_np[j][self.ai_index])
                real_prob = float(probs_np[j][self.real_index])
                
                prob_dict = {
                    self.ai_label: ai_prob,
                    self.real_label: real_prob
                }
                
                decision = self._make_decision(ai_prob)
                
                results.append(DetectionResult(
                    image_path=path,
                    probabilities=prob_dict,
                    ai_probability=ai_prob,
                    real_probability=real_prob,
                    decision=decision,
                    ai_label=self.ai_label,
                    real_label=self.real_label
                ))
        
        return results
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model configuration details
        """
        return {
            "model_id": MODEL_ID,
            "device": str(self.device),
            "labels": self.model.config.id2label,
            "ai_label": self.ai_label,
            "real_label": self.real_label,
            "thresholds": {
                "reject": self.thresholds.REJECT_THRESHOLD,
                "manual_review": self.thresholds.MANUAL_REVIEW_THRESHOLD
            }
        }


# =============================================================================
# Quick Test (when run directly)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI Image Detector - Quick Test")
    print("=" * 60)
    
    # Initialize detector
    detector = FoodImageDetector()
    
    # Print model info
    print("\n📋 Model Info:")
    info = detector.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Detector initialized successfully!")
    print("   Use detector.predict('image.jpg') to analyze an image")
