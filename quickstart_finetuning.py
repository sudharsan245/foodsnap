"""
Quick Start Script for Food-Specific AI Detection Fine-Tuning

This script provides an interactive setup and validation workflow.

Usage:
    python quickstart_finetuning.py
"""

import sys
from pathlib import Path
import subprocess


def print_header(text: str) -> None:
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    print_header("CHECKING DEPENDENCIES")
    
    required = [
        'torch',
        'torchvision',
        'transformers',
        'PIL',
        'sklearn',
        'matplotlib',
        'seaborn',
        'tensorboard',
        'tqdm'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✅ {package:20s} - installed")
        except ImportError:
            print(f"❌ {package:20s} - missing")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install -r requirements_finetuning.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def check_dataset_structure(data_dir: str) -> bool:
    """Check if dataset has correct structure."""
    print_header("CHECKING DATASET STRUCTURE")
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        print("\nCreate the directory structure:")
        print(f"  {data_dir}/")
        print("  ├── real_clean/")
        print("  ├── real_contaminated/")
        print("  └── ai_generated/")
        return False
    
    required_dirs = ['real_clean', 'real_contaminated', 'ai_generated']
    all_exist = True
    
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if dir_path.exists():
            # Count images
            image_count = len(list(dir_path.glob('*.jpg'))) + \
                         len(list(dir_path.glob('*.jpeg'))) + \
                         len(list(dir_path.glob('*.png')))
            
            status = "✅" if image_count > 0 else "⚠️"
            print(f"{status} {dir_name:20s} - {image_count} images")
            
            if image_count == 0:
                all_exist = False
        else:
            print(f"❌ {dir_name:20s} - directory missing")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️ Dataset structure incomplete")
        print("\nSee FINETUNING_GUIDE.md for dataset preparation instructions")
        return False
    
    print("\n✅ Dataset structure looks good!")
    return True


def run_phase_1() -> bool:
    """Run Phase 1: Model Inspection."""
    print_header("PHASE 1: MODEL INSPECTION")
    
    print("This will inspect the SMOGY model and create a freezing configuration.")
    response = input("Run Phase 1? (y/n): ").strip().lower()
    
    if response != 'y':
        print("⏭️ Skipping Phase 1")
        return True
    
    try:
        subprocess.run([
            sys.executable, 'inspect_model.py', '--all'
        ], check=True)
        print("\n✅ Phase 1 complete!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Phase 1 failed")
        return False


def run_phase_3(data_dir: str) -> bool:
    """Run Phase 3: Fine-Tuning."""
    print_header("PHASE 3: FINE-TUNING")
    
    print("This will start the fine-tuning process.")
    print("\nRecommended settings:")
    print("  - Epochs: 15")
    print("  - Batch size: 16 (adjust based on GPU memory)")
    print("  - Learning rate: 1e-5")
    
    response = input("\nRun Phase 3 with default settings? (y/n): ").strip().lower()
    
    if response != 'y':
        print("⏭️ Skipping Phase 3")
        print("\nTo run manually:")
        print(f"  python finetune.py --data_dir {data_dir} --epochs 15 --batch_size 16")
        return True
    
    try:
        subprocess.run([
            sys.executable, 'finetune.py',
            '--data_dir', data_dir,
            '--epochs', '15',
            '--batch_size', '16',
            '--learning_rate', '1e-5',
            '--output_dir', './checkpoints'
        ], check=True)
        print("\n✅ Phase 3 complete!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Phase 3 failed")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return False


def run_phase_5(test_dir: str) -> bool:
    """Run Phase 5: Evaluation."""
    print_header("PHASE 5: EVALUATION")
    
    checkpoint_path = Path('./checkpoints/best_model.pth')
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("\nTrain the model first (Phase 3)")
        return False
    
    print(f"Found checkpoint: {checkpoint_path}")
    response = input("\nRun evaluation? (y/n): ").strip().lower()
    
    if response != 'y':
        print("⏭️ Skipping Phase 5")
        print("\nTo run manually:")
        print(f"  python evaluate.py --model_path {checkpoint_path} --data_dir {test_dir}")
        return True
    
    try:
        subprocess.run([
            sys.executable, 'evaluate.py',
            '--model_path', str(checkpoint_path),
            '--data_dir', test_dir,
            '--output_dir', './evaluation_results'
        ], check=True)
        print("\n✅ Phase 5 complete!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Phase 5 failed")
        return False


def main():
    """Main quickstart workflow."""
    print_header("FOOD-SPECIFIC AI DETECTION - FINE-TUNING QUICKSTART")
    
    print("This script will guide you through the fine-tuning process.")
    print("\nPhases:")
    print("  1. Model Inspection (inspect architecture, create freezing config)")
    print("  2. Dataset Integration (handled by training script)")
    print("  3. Fine-Tuning (train the model)")
    print("  4. Hard Negatives (manual - add to dataset and retrain)")
    print("  5. Evaluation (test the model)")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install dependencies first")
        sys.exit(1)
    
    # Get dataset directory
    print("\n" + "-" * 80)
    data_dir = input("Enter path to training dataset (default: ./food_ai_dataset): ").strip()
    if not data_dir:
        data_dir = './food_ai_dataset'
    
    # Check dataset structure
    if not check_dataset_structure(data_dir):
        print("\n❌ Please prepare the dataset first")
        print("See FINETUNING_GUIDE.md for instructions")
        sys.exit(1)
    
    # Phase 1: Model Inspection
    if not run_phase_1():
        print("\n❌ Quickstart failed at Phase 1")
        sys.exit(1)
    
    # Phase 3: Fine-Tuning
    if not run_phase_3(data_dir):
        print("\n⚠️ Training incomplete")
        print("\nYou can resume training with:")
        print("  python finetune.py --data_dir {} --resume ./checkpoints/checkpoint_epoch_X.pth".format(data_dir))
        sys.exit(1)
    
    # Phase 5: Evaluation
    print("\n" + "-" * 80)
    test_dir = input("Enter path to test dataset (default: ./test_data): ").strip()
    if not test_dir:
        test_dir = './test_data'
    
    if Path(test_dir).exists():
        run_phase_5(test_dir)
    else:
        print(f"\n⚠️ Test directory not found: {test_dir}")
        print("Skipping evaluation")
    
    # Summary
    print_header("QUICKSTART COMPLETE")
    
    print("✅ Fine-tuning workflow completed!")
    print("\nNext steps:")
    print("  1. Review evaluation results in ./evaluation_results/")
    print("  2. Check TensorBoard logs: tensorboard --logdir ./checkpoints/logs")
    print("  3. Update config.py to use fine-tuned model")
    print("  4. Test with Flask app: python app.py")
    print("\nFor detailed instructions, see FINETUNING_GUIDE.md")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Quickstart interrupted by user")
        sys.exit(0)
