"""
PHASE 2: Food-Specific Dataset Integration

This module provides dataset handling for food-specific AI detection fine-tuning.

Dataset Classes:
    - Class 0: Real Food (Clean)
    - Class 1: Real Food (Contaminated)
    - Class 2: AI-Generated Food

Features:
    - Heavy real-world augmentations (JPEG compression, blur, low-light, etc.)
    - Class balancing
    - Hard negative examples support
    - Compatible with PyTorch DataLoader

Usage:
    from dataset import FoodAIDataset, create_dataloaders
    
    train_loader, val_loader = create_dataloaders(
        data_dir='path/to/dataset',
        batch_size=16
    )
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import random
import io


class JPEGCompressionTransform:
    """Simulate JPEG compression artifacts."""
    
    def __init__(self, quality_range: Tuple[int, int] = (60, 95)):
        self.quality_range = quality_range
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random JPEG compression."""
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')


class LowLightTransform:
    """Simulate low-light conditions."""
    
    def __init__(self, brightness_range: Tuple[float, float] = (0.5, 1.5)):
        self.brightness_range = brightness_range
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random brightness adjustment."""
        from PIL import ImageEnhance
        factor = random.uniform(*self.brightness_range)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)


class FoodAIDataset(Dataset):
    """
    Dataset for food-specific AI image detection.
    
    Expected directory structure:
        data_dir/
        ├── real_clean/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── real_contaminated/
        │   ├── hair_1.jpg
        │   ├── insect_1.jpg
        │   ├── mold_1.jpg
        │   └── ...
        └── ai_generated/
            ├── ai_clean_1.jpg
            ├── ai_contaminated_1.jpg
            └── ...
    
    Classes:
        0: Real Food (Clean)
        1: Real Food (Contaminated)
        2: AI-Generated Food
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: int = 224,
        augment: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing class subdirectories
            split: 'train', 'val', or 'test'
            image_size: Target image size (default: 224 for Swin Transformer)
            augment: Whether to apply augmentations (only for training)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        
        # Class mapping
        self.class_to_idx = {
            'real_clean': 0,
            'real_contaminated': 1,
            'ai_generated': 2
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        # Setup transforms
        self.transform = self._get_transforms()
        
        print(f"📊 Loaded {len(self.samples)} images for {split} split")
        self._print_class_distribution()
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all image paths and their labels."""
        samples = []
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                print(f"⚠️ Warning: Directory not found: {class_dir}")
                continue
            
            # Find all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
            for ext in image_extensions:
                for img_path in class_dir.glob(ext):
                    samples.append((img_path, class_idx))
                for img_path in class_dir.glob(ext.upper()):
                    samples.append((img_path, class_idx))
        
        if len(samples) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
        
        return samples
    
    def _print_class_distribution(self) -> None:
        """Print class distribution statistics."""
        class_counts = {i: 0 for i in range(3)}
        for _, label in self.samples:
            class_counts[label] += 1
        
        total = len(self.samples)
        print(f"\n  Class Distribution ({self.split}):")
        for idx, count in class_counts.items():
            class_name = self.idx_to_class[idx]
            percentage = count / total * 100 if total > 0 else 0
            print(f"    {idx} ({class_name:20s}): {count:5d} ({percentage:5.1f}%)")
    
    def _get_transforms(self) -> transforms.Compose:
        """Get image transforms based on split and augmentation settings."""
        
        if self.augment:
            # Heavy augmentation for training
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1)
                ),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                ),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
                ], p=0.3),
                LowLightTransform(brightness_range=(0.5, 1.5)),
                JPEGCompressionTransform(quality_range=(60, 95)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # Minimal transforms for validation/test
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (self.image_size, self.image_size), color='black')
        
        # Apply transforms
        image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced training.
        
        Returns:
            Tensor of class weights (inverse frequency)
        """
        class_counts = torch.zeros(3)
        for _, label in self.samples:
            class_counts[label] += 1
        
        # Inverse frequency weighting
        total = len(self.samples)
        class_weights = total / (3 * class_counts)
        
        return class_weights


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 224,
    val_split: float = 0.2,
    balance_classes: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Root directory containing class subdirectories
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        image_size: Target image size
        val_split: Fraction of data to use for validation
        balance_classes: Whether to use weighted sampling for class balance
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from sklearn.model_selection import train_test_split
    
    # Load full dataset
    full_dataset = FoodAIDataset(
        data_dir=data_dir,
        split='full',
        image_size=image_size,
        augment=False
    )
    
    # Split into train/val
    indices = list(range(len(full_dataset)))
    labels = [label for _, label in full_dataset.samples]
    
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        stratify=labels,
        random_state=42
    )
    
    # Create train dataset with augmentation
    train_dataset = FoodAIDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        augment=True
    )
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
    
    # Create val dataset without augmentation
    val_dataset = FoodAIDataset(
        data_dir=data_dir,
        split='val',
        image_size=image_size,
        augment=False
    )
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices]
    
    # Create samplers for balanced training
    if balance_classes:
        # Calculate sample weights
        class_weights = train_dataset.get_class_weights()
        sample_weights = [class_weights[label] for _, label in train_dataset.samples]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✅ Dataloaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def create_test_dataloader(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 224
) -> DataLoader:
    """
    Create test dataloader.
    
    Args:
        data_dir: Root directory containing class subdirectories
        batch_size: Batch size for testing
        num_workers: Number of worker processes
        image_size: Target image size
    
    Returns:
        Test DataLoader
    """
    test_dataset = FoodAIDataset(
        data_dir=data_dir,
        split='test',
        image_size=image_size,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✅ Test dataloader created:")
    print(f"   Test batches: {len(test_loader)}")
    
    return test_loader


if __name__ == '__main__':
    """Test dataset loading."""
    print("=" * 80)
    print("PHASE 2: Dataset Loading Test")
    print("=" * 80)
    
    # Example usage
    print("\n📋 To use this dataset:")
    print("   1. Organize your data in the following structure:")
    print("      data/")
    print("      ├── real_clean/")
    print("      ├── real_contaminated/")
    print("      └── ai_generated/")
    print("\n   2. Create dataloaders:")
    print("      from dataset import create_dataloaders")
    print("      train_loader, val_loader = create_dataloaders('data/', batch_size=16)")
    print("\n   3. Iterate over batches:")
    print("      for images, labels in train_loader:")
    print("          # Training code here")
    print("          pass")
