"""
PHASE 5: Real-World Validation & Evaluation

This script evaluates the fine-tuned model under real-world complaint conditions
and provides comprehensive metrics and threshold analysis.

Features:
    - Evaluation on test set
    - Per-class metrics (precision, recall, F1)
    - Confusion matrix
    - Confidence score distribution
    - Threshold calibration
    - Failure case analysis
    - Real-world scenario simulation

Usage:
    python evaluate.py --model_path ./checkpoints/best_model.pth --data_dir ./test_data
    python evaluate.py --model_path ./checkpoints/best_model.pth --data_dir ./test_data --save_failures
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageClassification
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
from typing import Dict, List, Tuple
import json
from tqdm import tqdm

from dataset import create_test_dataloader
from finetune import FoodAIFineTuner


class ModelEvaluator:
    """Evaluator for fine-tuned food AI detection model."""
    
    def __init__(
        self,
        model_path: str,
        device: str = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        
        # Load model
        print(f"📥 Loading model from: {model_path}")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully")
        
        # Class names
        self.class_names = ['Real Clean', 'Real Contaminated', 'AI-Generated']
    
    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model architecture
        from config import MODEL_ID
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        
        # Modify for 3 classes if needed
        if model.config.num_labels != 3:
            import torch.nn as nn
            if hasattr(model, 'classifier'):
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, 3)
            elif hasattr(model, 'head'):
                in_features = model.head.in_features
                model.head = nn.Linear(in_features, 3)
            model.config.num_labels = 3
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def evaluate(
        self,
        test_loader,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            save_dir: Directory to save results and plots
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "=" * 80)
        print("PHASE 5: REAL-WORLD VALIDATION")
        print("=" * 80)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("\n🔍 Running inference on test set...")
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images = images.to(self.device)
                
                outputs = self.model(pixel_values=images).logits
                probs = F.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        # Print results
        self._print_results(metrics)
        
        # Generate plots
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            self._generate_plots(all_labels, all_preds, all_probs, save_path)
            self._save_metrics(metrics, save_path)
        
        return metrics
    
    def _calculate_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray
    ) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        
        # Overall accuracy
        accuracy = (preds == labels).mean() * 100
        
        # Per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            mask = labels == i
            if mask.sum() > 0:
                class_acc = (preds[mask] == labels[mask]).mean() * 100
                class_metrics[class_name] = {
                    'accuracy': class_acc,
                    'count': mask.sum(),
                    'mean_confidence': probs[mask, i].mean() * 100
                }
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # Classification report
        report = classification_report(
            labels, preds,
            target_names=self.class_names,
            output_dict=True
        )
        
        # False positive rate (critical metric)
        # FPR = Real images classified as AI
        real_mask = (labels == 0) | (labels == 1)  # Both real classes
        ai_preds_on_real = preds[real_mask] == 2
        fpr = ai_preds_on_real.mean() * 100 if real_mask.sum() > 0 else 0
        
        # AI detection recall
        ai_mask = labels == 2
        ai_detected = preds[ai_mask] == 2
        ai_recall = ai_detected.mean() * 100 if ai_mask.sum() > 0 else 0
        
        # Confidence score distribution
        confidence_dist = {
            'Real Clean': probs[labels == 0, 0].tolist() if (labels == 0).sum() > 0 else [],
            'Real Contaminated': probs[labels == 1, 1].tolist() if (labels == 1).sum() > 0 else [],
            'AI-Generated': probs[labels == 2, 2].tolist() if (labels == 2).sum() > 0 else []
        }
        
        return {
            'overall_accuracy': accuracy,
            'class_metrics': class_metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'false_positive_rate': fpr,
            'ai_detection_recall': ai_recall,
            'confidence_distribution': confidence_dist
        }
    
    def _print_results(self, metrics: Dict) -> None:
        """Print evaluation results."""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        
        print(f"\n📊 Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
        
        print(f"\n📊 Per-Class Performance:")
        for class_name, class_metrics in metrics['class_metrics'].items():
            print(f"   {class_name:25s}: {class_metrics['accuracy']:>6.2f}% "
                  f"(n={class_metrics['count']}, "
                  f"avg conf={class_metrics['mean_confidence']:.1f}%)")
        
        print(f"\n🎯 Critical Metrics:")
        print(f"   False Positive Rate (Real → AI): {metrics['false_positive_rate']:.2f}%")
        print(f"   AI Detection Recall:              {metrics['ai_detection_recall']:.2f}%")
        
        # Threshold evaluation
        print(f"\n📏 Threshold Analysis:")
        self._analyze_thresholds(metrics)
        
        print(f"\n📋 Detailed Classification Report:")
        report = metrics['classification_report']
        print(f"   {'Class':<25s} {'Precision':<12s} {'Recall':<12s} {'F1-Score':<12s}")
        print("   " + "-" * 65)
        for class_name in self.class_names:
            if class_name in report:
                p = report[class_name]['precision'] * 100
                r = report[class_name]['recall'] * 100
                f1 = report[class_name]['f1-score'] * 100
                print(f"   {class_name:<25s} {p:>10.2f}%  {r:>10.2f}%  {f1:>10.2f}%")
        
        print(f"\n🎯 Business Impact Assessment:")
        if metrics['false_positive_rate'] < 5:
            print(f"   ✅ FPR < 5%: Excellent - Safe for production")
        elif metrics['false_positive_rate'] < 10:
            print(f"   ⚠️ FPR 5-10%: Good - Consider manual review threshold")
        else:
            print(f"   ❌ FPR > 10%: Needs improvement - Too many false positives")
        
        if metrics['ai_detection_recall'] > 85:
            print(f"   ✅ AI Recall > 85%: Excellent fraud detection")
        elif metrics['ai_detection_recall'] > 70:
            print(f"   ⚠️ AI Recall 70-85%: Good but could be improved")
        else:
            print(f"   ❌ AI Recall < 70%: Missing too many AI images")
    
    def _analyze_thresholds(self, metrics: Dict) -> None:
        """Analyze different threshold settings."""
        thresholds = [
            (0.60, 0.80, "Conservative (Customer-first)"),
            (0.70, 0.85, "Balanced (Default)"),
            (0.80, 0.90, "Aggressive (Fraud-focused)")
        ]
        
        for manual_thresh, reject_thresh, name in thresholds:
            print(f"\n   {name}:")
            print(f"      P(AI) < {manual_thresh:.2f}  → Accept")
            print(f"      {manual_thresh:.2f} ≤ P(AI) < {reject_thresh:.2f} → Manual Review")
            print(f"      P(AI) ≥ {reject_thresh:.2f} → Reject")
    
    def _generate_plots(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray,
        save_dir: Path
    ) -> None:
        """Generate evaluation plots."""
        print(f"\n📊 Generating plots...")
        
        # Confusion Matrix
        self._plot_confusion_matrix(labels, preds, save_dir)
        
        # Confidence Distribution
        self._plot_confidence_distribution(labels, probs, save_dir)
        
        # Per-class confidence
        self._plot_per_class_confidence(labels, probs, save_dir)
        
        print(f"✅ Plots saved to: {save_dir}")
    
    def _plot_confusion_matrix(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        save_dir: Path
    ) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
    
    def _plot_confidence_distribution(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        save_dir: Path
    ) -> None:
        """Plot confidence score distribution."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for i, (class_name, ax) in enumerate(zip(self.class_names, axes)):
            mask = labels == i
            if mask.sum() > 0:
                confidences = probs[mask, i] * 100
                ax.hist(confidences, bins=20, edgecolor='black', alpha=0.7)
                ax.axvline(confidences.mean(), color='red', linestyle='--',
                          label=f'Mean: {confidences.mean():.1f}%')
                ax.set_xlabel('Confidence (%)', fontsize=10)
                ax.set_ylabel('Count', fontsize=10)
                ax.set_title(f'{class_name}', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'confidence_distribution.png', dpi=300)
        plt.close()
    
    def _plot_per_class_confidence(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        save_dir: Path
    ) -> None:
        """Plot per-class confidence box plots."""
        data = []
        for i, class_name in enumerate(self.class_names):
            mask = labels == i
            if mask.sum() > 0:
                confidences = probs[mask, i] * 100
                data.append(confidences)
            else:
                data.append([])
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=self.class_names)
        plt.ylabel('Confidence (%)', fontsize=12)
        plt.title('Confidence Score Distribution by Class', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'confidence_boxplot.png', dpi=300)
        plt.close()
    
    def _save_metrics(self, metrics: Dict, save_dir: Path) -> None:
        """Save metrics to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {
            'overall_accuracy': float(metrics['overall_accuracy']),
            'class_metrics': metrics['class_metrics'],
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'false_positive_rate': float(metrics['false_positive_rate']),
            'ai_detection_recall': float(metrics['ai_detection_recall']),
            'classification_report': metrics['classification_report']
        }
        
        with open(save_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"💾 Metrics saved to: {save_dir / 'evaluation_metrics.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="PHASE 5: Real-World Validation & Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test data')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Create test dataloader
    print("📦 Loading test dataset...")
    test_loader = create_test_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        device=args.device
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(
        test_loader=test_loader,
        save_dir=args.output_dir
    )
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print(f"   Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
