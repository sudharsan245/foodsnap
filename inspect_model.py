"""
PHASE 1: Model Inspection & Freezing Strategy

This script inspects the SMOGY Swin Transformer architecture and implements
the layer freezing strategy for domain adaptation.

Usage:
    python inspect_model.py --inspect    # View model architecture
    python inspect_model.py --freeze     # Show freezing plan
    python inspect_model.py --apply      # Apply freezing and save config
"""

import torch
import argparse
from transformers import AutoModelForImageClassification
from config import MODEL_ID
from typing import Dict, List, Tuple


class ModelInspector:
    """Inspector for SMOGY model architecture and parameter analysis."""
    
    def __init__(self, model_id: str = MODEL_ID):
        """Load model for inspection."""
        print(f"🔍 Loading model: {model_id}")
        self.model = AutoModelForImageClassification.from_pretrained(model_id)
        self.model_id = model_id
        print("✅ Model loaded successfully!\n")
    
    def inspect_architecture(self) -> None:
        """Display detailed model architecture."""
        print("=" * 80)
        print("MODEL ARCHITECTURE INSPECTION")
        print("=" * 80)
        
        print(f"\n📋 Model Type: {self.model.config.model_type}")
        print(f"📋 Architecture: {self.model.config.architectures}")
        print(f"📋 Number of Labels: {self.model.config.num_labels}")
        print(f"📋 Label Mapping: {self.model.config.id2label}")
        
        # Count total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n📊 Total Parameters: {total_params:,}")
        print(f"📊 Trainable Parameters: {trainable_params:,}")
        print(f"📊 Parameter Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
        
        # Display layer structure
        print("\n" + "=" * 80)
        print("LAYER STRUCTURE")
        print("=" * 80)
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    print(f"  {name:60s} | {num_params:>12,} params")
    
    def analyze_swin_structure(self) -> Dict[str, List[str]]:
        """
        Analyze Swin Transformer structure and categorize layers.
        
        Returns:
            Dictionary mapping layer categories to parameter names
        """
        print("\n" + "=" * 80)
        print("SWIN TRANSFORMER STRUCTURE ANALYSIS")
        print("=" * 80)
        
        layer_categories = {
            'patch_embed': [],
            'stage_1': [],
            'stage_2': [],
            'stage_3': [],
            'stage_4': [],
            'classification_head': []
        }
        
        for name, param in self.model.named_parameters():
            # Categorize based on naming patterns
            if 'embeddings' in name or 'patch_embed' in name:
                layer_categories['patch_embed'].append(name)
            elif 'encoder.layers.0' in name or 'layers.0' in name:
                layer_categories['stage_1'].append(name)
            elif 'encoder.layers.1' in name or 'layers.1' in name:
                layer_categories['stage_2'].append(name)
            elif 'encoder.layers.2' in name or 'layers.2' in name:
                layer_categories['stage_3'].append(name)
            elif 'encoder.layers.3' in name or 'layers.3' in name:
                layer_categories['stage_4'].append(name)
            elif 'classifier' in name or 'head' in name:
                layer_categories['classification_head'].append(name)
            else:
                # Fallback: try to detect stage from layer number
                if 'layer' in name:
                    # This is a heuristic, adjust based on actual model structure
                    layer_categories['stage_4'].append(name)
        
        # Print summary
        for category, params in layer_categories.items():
            param_count = sum(
                self.model.state_dict()[p].numel() 
                for p in params if p in self.model.state_dict()
            )
            print(f"\n{category.upper()}:")
            print(f"  Parameters: {len(params)}")
            print(f"  Total params: {param_count:,}")
            if len(params) > 0 and len(params) <= 5:
                for p in params:
                    print(f"    - {p}")
            elif len(params) > 5:
                print(f"    - {params[0]}")
                print(f"    - ...")
                print(f"    - {params[-1]}")
        
        return layer_categories
    
    def create_freezing_plan(self) -> Tuple[List[str], List[str]]:
        """
        Create layer freezing plan.
        
        Returns:
            Tuple of (layers_to_freeze, layers_to_train)
        """
        print("\n" + "=" * 80)
        print("LAYER FREEZING PLAN")
        print("=" * 80)
        
        layer_categories = self.analyze_swin_structure()
        
        # Define freezing strategy
        freeze_categories = ['patch_embed', 'stage_1', 'stage_2', 'stage_3']
        train_categories = ['stage_4', 'classification_head']
        
        layers_to_freeze = []
        layers_to_train = []
        
        for category in freeze_categories:
            layers_to_freeze.extend(layer_categories[category])
        
        for category in train_categories:
            layers_to_train.extend(layer_categories[category])
        
        # Calculate statistics
        freeze_params = sum(
            self.model.state_dict()[p].numel() 
            for p in layers_to_freeze if p in self.model.state_dict()
        )
        train_params = sum(
            self.model.state_dict()[p].numel() 
            for p in layers_to_train if p in self.model.state_dict()
        )
        total_params = freeze_params + train_params
        
        print("\n📊 FREEZING STATISTICS:")
        print(f"  Frozen Parameters:    {freeze_params:>12,} ({freeze_params/total_params*100:>5.1f}%)")
        print(f"  Trainable Parameters: {train_params:>12,} ({train_params/total_params*100:>5.1f}%)")
        print(f"  Total Parameters:     {total_params:>12,}")
        
        print("\n🔒 LAYERS TO FREEZE:")
        print(f"  Categories: {', '.join(freeze_categories)}")
        print(f"  Total layers: {len(layers_to_freeze)}")
        
        print("\n🔓 LAYERS TO TRAIN:")
        print(f"  Categories: {', '.join(train_categories)}")
        print(f"  Total layers: {len(layers_to_train)}")
        
        print("\n💡 RATIONALE:")
        print("  ✅ Freeze early layers: Preserve low-level feature extraction")
        print("  ✅ Freeze mid layers: Preserve texture/pattern detection")
        print("  ✅ Freeze stage 3: Preserve object understanding")
        print("  🔓 Train stage 4: Adapt to food-specific high-level features")
        print("  🔓 Train classifier: Learn food-specific decision boundaries")
        
        return layers_to_freeze, layers_to_train
    
    def apply_freezing(self, layers_to_freeze: List[str]) -> None:
        """
        Apply freezing to specified layers.
        
        Args:
            layers_to_freeze: List of parameter names to freeze
        """
        print("\n" + "=" * 80)
        print("APPLYING LAYER FREEZING")
        print("=" * 80)
        
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if name in layers_to_freeze:
                param.requires_grad = False
                frozen_count += 1
        
        # Verify freezing
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n✅ Freezing Applied!")
        print(f"  Frozen layers: {frozen_count}")
        print(f"  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    def save_freezing_config(self, layers_to_freeze: List[str], layers_to_train: List[str]) -> None:
        """Save freezing configuration to file."""
        import json
        from pathlib import Path
        
        config = {
            'model_id': self.model_id,
            'freezing_strategy': {
                'freeze_categories': ['patch_embed', 'stage_1', 'stage_2', 'stage_3'],
                'train_categories': ['stage_4', 'classification_head']
            },
            'layers_to_freeze': layers_to_freeze,
            'layers_to_train': layers_to_train,
            'statistics': {
                'total_layers': len(layers_to_freeze) + len(layers_to_train),
                'frozen_layers': len(layers_to_freeze),
                'trainable_layers': len(layers_to_train)
            }
        }
        
        output_path = Path(__file__).parent / 'freezing_config.json'
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Freezing configuration saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PHASE 1: Model Inspection & Freezing Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Inspect model architecture and parameters'
    )
    
    parser.add_argument(
        '--freeze',
        action='store_true',
        help='Show layer freezing plan'
    )
    
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply freezing and save configuration'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all inspections (inspect + freeze + apply)'
    )
    
    args = parser.parse_args()
    
    if not any([args.inspect, args.freeze, args.apply, args.all]):
        parser.print_help()
        print("\n❌ Please specify at least one action: --inspect, --freeze, --apply, or --all")
        return
    
    # Initialize inspector
    inspector = ModelInspector()
    
    # Run requested operations
    if args.inspect or args.all:
        inspector.inspect_architecture()
    
    if args.freeze or args.apply or args.all:
        layers_to_freeze, layers_to_train = inspector.create_freezing_plan()
    
    if args.apply or args.all:
        inspector.apply_freezing(layers_to_freeze)
        inspector.save_freezing_config(layers_to_freeze, layers_to_train)
    
    print("\n" + "=" * 80)
    print("✅ PHASE 1 COMPLETE")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Review the freezing configuration in freezing_config.json")
    print("  2. Proceed to PHASE 2: Dataset preparation (dataset.py)")
    print("  3. Then PHASE 3: Fine-tuning (finetune.py)")


if __name__ == '__main__':
    main()
