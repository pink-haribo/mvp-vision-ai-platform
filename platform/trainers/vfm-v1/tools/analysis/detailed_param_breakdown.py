#!/usr/bin/env python
"""
Detailed parameter breakdown by module.

Usage:
    python tools/analysis/detailed_param_breakdown.py CONFIG_FILE
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

import torch
from mmengine.config import Config
from mmengine.registry import DefaultScope

# Import mmyolo modules
from mmyolo.utils.setup_env import register_all_modules as register_mmyolo
from mmyolo.registry import MODELS

# Import YOLO-World modules to register components
import yolo_world

# Register all modules
register_mmyolo(init_default_scope=True)


def analyze_module_params(model, module_name=""):
    """Recursively analyze parameters by module."""
    results = {}
    
    for name, module in model.named_children():
        full_name = f"{module_name}.{name}" if module_name else name
        
        # Count params in this module
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = total - trainable
        
        if total > 0:
            results[full_name] = {
                'total': total,
                'trainable': trainable,
                'frozen': frozen,
                'ratio': trainable / total * 100 if total > 0 else 0
            }
        
        # Recursively analyze children
        child_results = analyze_module_params(module, full_name)
        results.update(child_results)
    
    return results


def print_breakdown(config_path):
    """Print detailed parameter breakdown."""
    print(f"\n{'='*100}")
    print(f"Config: {config_path}")
    print(f"{'='*100}\n")
    
    try:
        # Set default scope
        DefaultScope.get_instance('mmyolo', scope_name='mmyolo')
        
        # Load config
        cfg = Config.fromfile(config_path)
        
        # Build model
        print("Building model...")
        model = MODELS.build(cfg.model)
        model.eval()
        
        # Get top-level breakdown
        print("\n" + "="*100)
        print("TOP-LEVEL MODULE BREAKDOWN")
        print("="*100)
        print(f"{'Module':<40} {'Total':>15} {'Trainable':>15} {'Frozen':>15} {'Train %':>10}")
        print("-"*100)
        
        top_level = {}
        for name, module in model.named_children():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            frozen = total - trainable
            ratio = trainable / total * 100 if total > 0 else 0
            
            top_level[name] = {
                'total': total,
                'trainable': trainable,
                'frozen': frozen,
                'ratio': ratio
            }
            
            print(f"{name:<40} {total:>15,} {trainable:>15,} {frozen:>15,} {ratio:>9.1f}%")
        
        # Total
        total_all = sum(p.numel() for p in model.parameters())
        trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_all = total_all - trainable_all
        ratio_all = trainable_all / total_all * 100 if total_all > 0 else 0
        
        print("-"*100)
        print(f"{'TOTAL':<40} {total_all:>15,} {trainable_all:>15,} {frozen_all:>15,} {ratio_all:>9.1f}%")
        
        # Detailed backbone breakdown
        if hasattr(model, 'backbone'):
            print("\n" + "="*100)
            print("BACKBONE DETAILED BREAKDOWN")
            print("="*100)
            print(f"{'Module':<50} {'Total':>15} {'Trainable':>15} {'Frozen':>15} {'Train %':>10}")
            print("-"*100)
            
            for name, module in model.backbone.named_children():
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                frozen = total - trainable
                ratio = trainable / total * 100 if total > 0 else 0
                
                print(f"{name:<50} {total:>15,} {trainable:>15,} {frozen:>15,} {ratio:>9.1f}%")
                
                # If this is image_model or text_model, show more detail
                if name in ['image_model', 'text_model']:
                    for sub_name, sub_module in module.named_children():
                        sub_total = sum(p.numel() for p in sub_module.parameters())
                        sub_trainable = sum(p.numel() for p in sub_module.parameters() if p.requires_grad)
                        sub_frozen = sub_total - sub_trainable
                        sub_ratio = sub_trainable / sub_total * 100 if sub_total > 0 else 0
                        
                        if sub_total > 0:
                            print(f"  └─ {sub_name:<46} {sub_total:>15,} {sub_trainable:>15,} {sub_frozen:>15,} {sub_ratio:>9.1f}%")
        
        # Neck breakdown
        if hasattr(model, 'neck'):
            print("\n" + "="*100)
            print("NECK DETAILED BREAKDOWN")
            print("="*100)
            print(f"{'Module':<50} {'Total':>15} {'Trainable':>15} {'Frozen':>15} {'Train %':>10}")
            print("-"*100)
            
            for name, module in model.neck.named_children():
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                frozen = total - trainable
                ratio = trainable / total * 100 if total > 0 else 0
                
                if total > 0:
                    print(f"{name:<50} {total:>15,} {trainable:>15,} {frozen:>15,} {ratio:>9.1f}%")
        
        # Head breakdown
        if hasattr(model, 'bbox_head'):
            print("\n" + "="*100)
            print("HEAD DETAILED BREAKDOWN")
            print("="*100)
            print(f"{'Module':<50} {'Total':>15} {'Trainable':>15} {'Frozen':>15} {'Train %':>10}")
            print("-"*100)
            
            for name, module in model.bbox_head.named_children():
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                frozen = total - trainable
                ratio = trainable / total * 100 if total > 0 else 0
                
                if total > 0:
                    print(f"{name:<50} {total:>15,} {trainable:>15,} {frozen:>15,} {ratio:>9.1f}%")
        
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)
        print(f"Total Parameters:      {total_all:>15,} ({total_all/1e6:>6.2f}M)")
        print(f"Trainable Parameters:  {trainable_all:>15,} ({trainable_all/1e6:>6.2f}M)")
        print(f"Frozen Parameters:     {frozen_all:>15,} ({frozen_all/1e6:>6.2f}M)")
        print(f"Trainable Ratio:       {ratio_all:>15.1f}%")
        print("="*100)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    configs = [
        'configs/finetune_coco/vfm_v1_l_mvtec.py',
    ]
    
    if len(sys.argv) > 1:
        configs = [sys.argv[1]]
    
    for config in configs:
        print_breakdown(config)

