"""
AI-Generated Image Detection - Main Entry Point

This script demonstrates how to use the FoodImageDetector for:
1. Single image inference
2. Batch inference on multiple images
3. Processing a directory of images

Usage:
    # Single image
    python main.py --image path/to/food.jpg
    
    # Multiple images
    python main.py --images img1.jpg img2.jpg img3.jpg
    
    # Directory of images
    python main.py --directory path/to/images/
    
    # Quick test (load model and show info)
    python main.py --test
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List

from detector import FoodImageDetector, DetectionResult
from config import ClaimDecision


def print_result(result: DetectionResult, verbose: bool = True) -> None:
    """Pretty print a detection result."""
    print("\n" + "=" * 60)
    print(f"📸 Image: {Path(result.image_path).name}")
    print("=" * 60)
    
    # Probabilities
    print(f"\n📊 Probabilities:")
    print(f"   {result.ai_label}: {result.ai_probability:.2%}")
    print(f"   {result.real_label}: {result.real_probability:.2%}")
    
    # Decision with visual indicator
    print(f"\n🎯 Decision: {result.decision.emoji} {result.decision.description}")
    
    if verbose:
        # Action recommendation
        print(f"\n📋 Recommended Action:")
        if result.decision == ClaimDecision.ACCEPT:
            print("   ✅ Process refund claim normally")
            print("   📝 Image appears to be a genuine photograph")
        elif result.decision == ClaimDecision.REJECT:
            print("   ❌ Block automatic refund")
            print("   🚨 Flag account for review")
            print("   📝 High probability of AI-generated image")
        else:  # MANUAL_REVIEW
            print("   ⏳ Queue for human reviewer")
            print("   📝 Confidence is not high enough for automatic decision")


def print_summary(results: List[DetectionResult]) -> None:
    """Print summary statistics for batch results."""
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return
    
    accept_count = sum(1 for r in results if r.decision == ClaimDecision.ACCEPT)
    reject_count = sum(1 for r in results if r.decision == ClaimDecision.REJECT)
    review_count = sum(1 for r in results if r.decision == ClaimDecision.MANUAL_REVIEW)
    
    print("\n" + "=" * 60)
    print("📊 BATCH SUMMARY")
    print("=" * 60)
    print(f"   Total images processed: {total}")
    print(f"   🟢 Accept:        {accept_count} ({accept_count/total:.1%})")
    print(f"   🔴 Reject:        {reject_count} ({reject_count/total:.1%})")
    print(f"   🟡 Manual Review: {review_count} ({review_count/total:.1%})")


def get_images_from_directory(directory: Path, extensions: tuple = ('.jpg', '.jpeg', '.png', '.webp')) -> List[Path]:
    """Get all image files from a directory."""
    images = []
    for ext in extensions:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(
        description="AI-Generated Image Detection for Food Delivery Fraud Prevention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --test                          # Test model loading
  python main.py --image food.jpg                # Analyze single image
  python main.py --images a.jpg b.jpg c.jpg      # Analyze multiple images
  python main.py --directory ./images/           # Analyze all images in directory
  python main.py --image food.jpg --json         # Output as JSON
        """
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: load model and display configuration"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single image file to analyze"
    )
    
    parser.add_argument(
        "--images",
        nargs="+",
        type=str,
        help="Paths to multiple image files to analyze"
    )
    
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to directory containing images to analyze"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Force specific device (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.test, args.image, args.images, args.directory]):
        parser.print_help()
        print("\n❌ Error: Please specify --test, --image, --images, or --directory")
        sys.exit(1)
    
    # Initialize detector
    print("🚀 Initializing AI Image Detector...")
    print("-" * 60)
    detector = FoodImageDetector(device=args.device)
    print("-" * 60)
    
    # Test mode
    if args.test:
        print("\n✅ Model loaded successfully!")
        print("\n📋 Model Configuration:")
        info = detector.get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        return
    
    # Collect images to process
    image_paths: List[str] = []
    
    if args.image:
        image_paths.append(args.image)
    
    if args.images:
        image_paths.extend(args.images)
    
    if args.directory:
        dir_path = Path(args.directory)
        if not dir_path.exists():
            print(f"❌ Error: Directory not found: {args.directory}")
            sys.exit(1)
        dir_images = get_images_from_directory(dir_path)
        image_paths.extend([str(p) for p in dir_images])
        print(f"📁 Found {len(dir_images)} images in {args.directory}")
    
    if not image_paths:
        print("❌ Error: No images to process")
        sys.exit(1)
    
    # Run inference
    print(f"\n🔍 Processing {len(image_paths)} image(s)...")
    
    if len(image_paths) == 1:
        # Single image - detailed output
        result = detector.predict(image_paths[0])
        
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print_result(result)
    else:
        # Batch processing
        results = detector.predict_batch(image_paths, batch_size=args.batch_size)
        
        if args.json:
            output = [r.to_dict() for r in results]
            print(json.dumps(output, indent=2))
        else:
            for result in results:
                print_result(result, verbose=False)
            print_summary(results)


if __name__ == "__main__":
    main()
