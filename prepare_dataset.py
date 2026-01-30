"""
Dataset Preparation Helper Script

This script helps organize and validate your dataset for fine-tuning.

Features:
- Create directory structure
- Validate image files
- Check class balance
- Generate dataset statistics
- Split data into train/val/test

Usage:
    python prepare_dataset.py --source ./raw_images --output ./food_ai_dataset
    python prepare_dataset.py --validate ./food_ai_dataset
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import random


def create_directory_structure(output_dir: str) -> None:
    """Create the required directory structure."""
    print("📁 Creating directory structure...")
    
    base_path = Path(output_dir)
    
    # Training directories
    (base_path / 'real_clean').mkdir(parents=True, exist_ok=True)
    (base_path / 'real_contaminated').mkdir(parents=True, exist_ok=True)
    (base_path / 'ai_generated').mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directories in: {output_dir}")
    print(f"   - {output_dir}/real_clean/")
    print(f"   - {output_dir}/real_contaminated/")
    print(f"   - {output_dir}/ai_generated/")


def validate_image(image_path: Path) -> bool:
    """Validate that a file is a valid image."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def scan_dataset(data_dir: str) -> Dict[str, List[Path]]:
    """Scan dataset and categorize images."""
    print(f"\n🔍 Scanning dataset: {data_dir}")
    
    data_path = Path(data_dir)
    dataset = {
        'real_clean': [],
        'real_contaminated': [],
        'ai_generated': []
    }
    
    for class_name in dataset.keys():
        class_dir = data_path / class_name
        if class_dir.exists():
            # Find all image files
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
            for ext in extensions:
                dataset[class_name].extend(class_dir.glob(ext))
                dataset[class_name].extend(class_dir.glob(ext.upper()))
    
    return dataset


def validate_dataset(data_dir: str) -> None:
    """Validate dataset structure and contents."""
    print("=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)
    
    dataset = scan_dataset(data_dir)
    
    # Statistics
    total_images = sum(len(images) for images in dataset.values())
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images: {total_images}")
    
    for class_name, images in dataset.items():
        count = len(images)
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"   {class_name:20s}: {count:5d} ({percentage:5.1f}%)")
    
    # Validate images
    print(f"\n🔍 Validating image files...")
    invalid_images = []
    
    for class_name, images in dataset.items():
        for img_path in images:
            if not validate_image(img_path):
                invalid_images.append(img_path)
    
    if invalid_images:
        print(f"\n⚠️ Found {len(invalid_images)} invalid images:")
        for img_path in invalid_images[:10]:  # Show first 10
            print(f"   - {img_path}")
        if len(invalid_images) > 10:
            print(f"   ... and {len(invalid_images) - 10} more")
    else:
        print(f"✅ All images are valid!")
    
    # Check balance
    print(f"\n⚖️ Class Balance Analysis:")
    counts = [len(images) for images in dataset.values()]
    if counts:
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 2.0:
            print(f"   ⚠️ Dataset is imbalanced (ratio: {imbalance_ratio:.2f})")
            print(f"   Consider balancing classes or using weighted sampling")
        else:
            print(f"   ✅ Dataset is reasonably balanced (ratio: {imbalance_ratio:.2f})")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    min_recommended = 300
    recommended = 1000
    
    for class_name, images in dataset.items():
        count = len(images)
        if count < min_recommended:
            print(f"   ⚠️ {class_name}: Need at least {min_recommended} images (have {count})")
        elif count < recommended:
            print(f"   📝 {class_name}: Recommended {recommended}+ images (have {count})")
        else:
            print(f"   ✅ {class_name}: Good sample size ({count} images)")


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> None:
    """Split dataset into train/val/test sets."""
    print("=" * 80)
    print("DATASET SPLITTING")
    print("=" * 80)
    
    random.seed(seed)
    
    # Scan source dataset
    dataset = scan_dataset(source_dir)
    
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in dataset.keys():
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Split each class
    for class_name, images in dataset.items():
        print(f"\n📂 Splitting {class_name}...")
        
        # Shuffle
        images = list(images)
        random.shuffle(images)
        
        # Calculate split indices
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # Copy files
        for split_name, split_images in [
            ('train', train_images),
            ('val', val_images),
            ('test', test_images)
        ]:
            dest_dir = output_path / split_name / class_name
            for img_path in split_images:
                shutil.copy2(img_path, dest_dir / img_path.name)
        
        print(f"   Train: {len(train_images)}")
        print(f"   Val:   {len(val_images)}")
        print(f"   Test:  {len(test_images)}")
    
    print(f"\n✅ Dataset split complete!")
    print(f"   Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Preparation Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--create',
        type=str,
        help='Create directory structure at specified path'
    )
    
    parser.add_argument(
        '--validate',
        type=str,
        help='Validate dataset at specified path'
    )
    
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split dataset into train/val/test'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        help='Source directory for splitting'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for split dataset'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    
    args = parser.parse_args()
    
    if args.create:
        create_directory_structure(args.create)
    
    elif args.validate:
        validate_dataset(args.validate)
    
    elif args.split:
        if not args.source or not args.output:
            print("❌ Error: --split requires --source and --output")
            return
        
        split_dataset(
            source_dir=args.source,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
    
    else:
        parser.print_help()
        print("\n❌ Please specify an action: --create, --validate, or --split")


if __name__ == '__main__':
    main()
