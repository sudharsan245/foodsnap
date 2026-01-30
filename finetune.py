"""
PHASE 3: Fine-Tuning Procedure

This script implements the fine-tuning procedure for adapting SMOGY to food-specific
AI detection with frozen layers and domain-specific training.

Features:
    - Layer freezing based on Phase 1 configuration
    - 3-class training (Real Clean, Real Contaminated, AI-Generated)
    - Low learning rate fine-tuning
    - Class-weighted loss
    - Early stopping
    - Checkpoint saving
    - TensorBoard logging

Usage:
    python finetune.py --data_dir ./data --epochs 15 --batch_size 16
    python finetune.py --data_dir ./data --resume ./checkpoints/best_model.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForImageClassification, AutoImageProcessor
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import numpy as np
from datetime import datetime

from config import MODEL_ID
from dataset import create_dataloaders, create_test_dataloader


class FoodAIFineTuner:
    """Fine-tuner for SMOGY model on food-specific AI detection."""
    
    def __init__(
        self,
        model_id: str = MODEL_ID,
        num_classes: int = 3,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
        freeze_config_path: Optional[str] = None
    ):
        """
        Initialize fine-tuner.
        
        Args:
            model_id: Hugging Face model ID or local path
            num_classes: Number of output classes (3 for food-specific)
            learning_rate: Learning rate for fine-tuning
            weight_decay: Weight decay for regularization
            device: Device to use ('cuda' or 'cpu')
            freeze_config_path: Path to freezing configuration JSON
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        
        # Load model
        print(f"📥 Loading model: {model_id}")
        self.model = AutoModelForImageClassification.from_pretrained(model_id)
        
        # Modify classification head for 3 classes if needed
        if self.model.config.num_labels != num_classes:
            print(f"🔄 Modifying classification head: {self.model.config.num_labels} → {num_classes} classes")
            self._modify_classification_head(num_classes)
        
        self.model.to(self.device)
        
        # Apply layer freezing
        if freeze_config_path and Path(freeze_config_path).exists():
            self._apply_freezing_from_config(freeze_config_path)
        else:
            print("⚠️ No freezing config found, freezing layers by heuristic")
            self._apply_default_freezing()
        
        # Setup optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        print(f"✅ Fine-tuner initialized")
        self._print_trainable_params()
    
    def _modify_classification_head(self, num_classes: int) -> None:
        """Modify the classification head for new number of classes."""
        # For Swin Transformer, the classifier is typically named 'classifier'
        if hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'head'):
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("Could not find classification head in model")
        
        # Update config
        self.model.config.num_labels = num_classes
    
    def _apply_freezing_from_config(self, config_path: str) -> None:
        """Apply layer freezing from configuration file."""
        print(f"📋 Loading freezing config from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        layers_to_freeze = config.get('layers_to_freeze', [])
        
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if name in layers_to_freeze:
                param.requires_grad = False
                frozen_count += 1
        
        print(f"✅ Frozen {frozen_count} layers from config")
    
    def _apply_default_freezing(self) -> None:
        """Apply default freezing strategy (freeze 80-90% of layers)."""
        print("🔒 Applying default freezing strategy...")
        
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze classification head
        if hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'head'):
            for param in self.model.head.parameters():
                param.requires_grad = True
        
        # Unfreeze last transformer stage (heuristic: last 20% of encoder layers)
        unfrozen_layers = 0
        for name, param in self.model.named_parameters():
            # This is a heuristic - adjust based on actual model structure
            if 'encoder.layers.3' in name or 'layers.3' in name:
                param.requires_grad = True
                unfrozen_layers += 1
        
        print(f"✅ Unfrozen {unfrozen_layers} encoder layers + classification head")
    
    def _print_trainable_params(self) -> None:
        """Print trainable parameter statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n📊 Parameter Statistics:")
        print(f"   Total:     {total_params:>12,} params")
        print(f"   Trainable: {trainable_params:>12,} params ({trainable_params/total_params*100:>5.1f}%)")
        print(f"   Frozen:    {frozen_params:>12,} params ({frozen_params/total_params*100:>5.1f}%)")
    
    def train_epoch(
        self,
        train_loader,
        criterion,
        epoch: int,
        writer: Optional[SummaryWriter] = None
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
            
            # TensorBoard logging
            if writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate(
        self,
        val_loader,
        criterion,
        epoch: int,
        writer: Optional[SummaryWriter] = None
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class accuracy
        class_correct = [0, 0, 0]
        class_total = [0, 0, 0]
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(pixel_values=images).logits
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Per-class statistics
                for i in range(3):
                    mask = labels == i
                    class_total[i] += mask.sum().item()
                    class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        # Per-class accuracy
        class_acc = [
            100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            for i in range(3)
        ]
        
        print(f"\n📊 Validation Results:")
        print(f"   Overall Accuracy: {epoch_acc:.2f}%")
        print(f"   Class 0 (Real Clean):        {class_acc[0]:.2f}%")
        print(f"   Class 1 (Real Contaminated): {class_acc[1]:.2f}%")
        print(f"   Class 2 (AI-Generated):      {class_acc[2]:.2f}%")
        
        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('Val/Loss', epoch_loss, epoch)
            writer.add_scalar('Val/Accuracy', epoch_acc, epoch)
            for i in range(3):
                writer.add_scalar(f'Val/Class{i}_Accuracy', class_acc[i], epoch)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'class_accuracies': class_acc
        }
    
    def save_checkpoint(self, filepath: str, metrics: Dict) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc
        }
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"✅ Checkpoint loaded from: {filepath}")
        print(f"   Resuming from epoch {self.current_epoch}")


def train(
    data_dir: str,
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    output_dir: str = './checkpoints',
    freeze_config: Optional[str] = None,
    resume: Optional[str] = None,
    early_stopping_patience: int = 5
):
    """
    Main training function.
    
    Args:
        data_dir: Directory containing training data
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Directory to save checkpoints
        freeze_config: Path to freezing configuration
        resume: Path to checkpoint to resume from
        early_stopping_patience: Epochs to wait before early stopping
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = output_path / 'logs' / timestamp
    writer = SummaryWriter(log_dir)
    
    print("=" * 80)
    print("PHASE 3: FINE-TUNING PROCEDURE")
    print("=" * 80)
    
    # Create dataloaders
    print("\n📦 Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        balance_classes=True
    )
    
    # Initialize fine-tuner
    print("\n🔧 Initializing fine-tuner...")
    finetuner = FoodAIFineTuner(
        learning_rate=learning_rate,
        freeze_config_path=freeze_config
    )
    
    # Resume from checkpoint if specified
    if resume:
        finetuner.load_checkpoint(resume)
    
    # Setup loss function with class weights
    class_weights = train_loader.dataset.get_class_weights().to(finetuner.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"\n📊 Class weights: {class_weights.cpu().numpy()}")
    
    # Training loop
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print("=" * 80)
    
    patience_counter = 0
    
    for epoch in range(finetuner.current_epoch + 1, epochs + 1):
        finetuner.current_epoch = epoch
        
        print(f"\n📅 Epoch {epoch}/{epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = finetuner.train_epoch(train_loader, criterion, epoch, writer)
        print(f"   Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        
        # Validate
        val_metrics = finetuner.validate(val_loader, criterion, epoch, writer)
        
        # Save best model
        if val_metrics['accuracy'] > finetuner.best_val_acc:
            finetuner.best_val_acc = val_metrics['accuracy']
            finetuner.best_val_loss = val_metrics['loss']
            best_path = output_path / 'best_model.pth'
            finetuner.save_checkpoint(best_path, val_metrics)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save latest checkpoint
        latest_path = output_path / f'checkpoint_epoch_{epoch}.pth'
        finetuner.save_checkpoint(latest_path, val_metrics)
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n⏹️ Early stopping triggered (patience={early_stopping_patience})")
            break
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"   Best Validation Accuracy: {finetuner.best_val_acc:.2f}%")
    print(f"   Best Validation Loss: {finetuner.best_val_loss:.4f}")
    print(f"   Checkpoints saved to: {output_path}")
    print(f"   TensorBoard logs: {log_dir}")
    print(f"\n   View logs with: tensorboard --logdir {log_dir}")
    
    writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="PHASE 3: Fine-Tuning Procedure",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs (default: 15)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate (default: 1e-5)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--freeze_config', type=str, default='./freezing_config.json',
                       help='Path to freezing configuration')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--early_stopping', type=int, default=5,
                       help='Early stopping patience (default: 5)')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        freeze_config=args.freeze_config,
        resume=args.resume,
        early_stopping_patience=args.early_stopping
    )


if __name__ == '__main__':
    main()
