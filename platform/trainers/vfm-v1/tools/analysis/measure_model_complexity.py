#!/usr/bin/env python
"""
Measure model complexity: params, FLOPs, and inference speed.

Usage:
    python tools/analysis/measure_model_complexity.py
"""

import sys
import time
import torch
import numpy as np

# Add current directory to path
sys.path.insert(0, '.')

# Import YOLO-World modules FIRST to register custom components
import yolo_world
from yolo_world.models.data_preprocessors import YOLOWDetDataPreprocessor

# Then import mmengine/mmyolo
from mmengine.config import Config
from mmengine.runner import Runner
from mmyolo.registry import MODELS
from mmengine.analysis import get_model_complexity_info


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def measure_inference_speed(model, input_shape=(1, 3, 640, 640), warmup=10, repeat=100, device='cuda'):
    """Measure inference speed (ms per batch)."""
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Measure
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(repeat):
            start = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)  # ms
    
    return np.mean(times), np.std(times)


def analyze_config(config_path, input_shape=(1, 3, 640, 640)):
    """Analyze a single config file."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {config_path}")
    print(f"{'='*80}")
    
    try:
        # Load config
        cfg = Config.fromfile(config_path)
        
        # Build model
        model = MODELS.build(cfg.model)
        model.eval()
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        trainable_ratio = trainable_params / total_params * 100
        
        print(f"\n📊 Parameters:")
        print(f"  Total:      {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable:  {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"  Frozen:     {total_params - trainable_params:,} ({(total_params - trainable_params)/1e6:.2f}M)")
        print(f"  Ratio:      {trainable_ratio:.2f}%")
        
        # Count LoRA parameters if exists
        lora_params = sum(p.numel() for n, p in model.named_parameters() 
                         if 'lora' in n.lower() and p.requires_grad)
        if lora_params > 0:
            print(f"  LoRA:       {lora_params:,} ({lora_params/1e6:.2f}M, {lora_params/trainable_params*100:.1f}% of trainable)")
        
        # Measure FLOPs
        print(f"\n⚡ FLOPs:")
        try:
            # For detection models, we need to use data_preprocessor
            analysis_result = get_model_complexity_info(
                model,
                input_shape=input_shape,
                show_table=False,
                show_arch=False
            )
            flops = analysis_result['flops']
            params_check = analysis_result['params']
            
            print(f"  FLOPs:      {flops:,} ({flops/1e9:.2f}G)")
            print(f"  Params:     {params_check:,} ({params_check/1e6:.2f}M) [verification]")
        except Exception as e:
            print(f"  ⚠️  FLOPs measurement failed: {e}")
            print(f"  Skipping FLOPs calculation...")
        
        # Measure inference speed
        print(f"\n🚀 Inference Speed (input shape: {input_shape}):")
        if torch.cuda.is_available():
            try:
                mean_time, std_time = measure_inference_speed(
                    model, 
                    input_shape=input_shape,
                    warmup=10,
                    repeat=100,
                    device='cuda'
                )
                print(f"  Mean:       {mean_time:.2f} ± {std_time:.2f} ms/batch")
                print(f"  FPS:        {1000/mean_time:.2f} images/sec")
            except Exception as e:
                print(f"  ⚠️  Speed measurement failed: {e}")
        else:
            print(f"  ⚠️  CUDA not available, skipping speed measurement")
        
        return {
            'config': config_path,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_ratio,
            'lora_params': lora_params if lora_params > 0 else 0,
            'flops': flops if 'flops' in locals() else None,
            'inference_time_ms': mean_time if 'mean_time' in locals() else None,
            'fps': 1000/mean_time if 'mean_time' in locals() else None,
        }
        
    except Exception as e:
        print(f"\n❌ Error analyzing {config_path}:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function."""
    configs = [
        ('Baseline', 'configs/finetune_coco/vfm_v1_l_mvtec.py'),
        ('Step 1: Dense LoRA', 'configs/adapter/phase2_dense_lora_v1.py'),
        ('Step 2: Hybrid Moderate', 'configs/adapter/phase2_hybrid_v1.py'),
        ('Step 3: Hybrid Aggressive', 'configs/adapter/phase2_hybrid_aggressive_v1.py'),
    ]
    
    results = []
    
    for name, config_path in configs:
        print(f"\n\n{'#'*80}")
        print(f"# {name}")
        print(f"{'#'*80}")
        
        result = analyze_config(config_path)
        if result:
            result['name'] = name
            results.append(result)
    
    # Print summary table
    print(f"\n\n{'='*100}")
    print(f"📊 SUMMARY TABLE")
    print(f"{'='*100}")
    
    print(f"\n{'Model':<30} {'Total Params':<15} {'Trainable':<15} {'Ratio':<10} {'LoRA':<12} {'FLOPs':<12} {'Speed (ms)':<12} {'FPS':<10}")
    print(f"{'-'*100}")
    
    for r in results:
        name = r['name']
        total = f"{r['total_params']/1e6:.2f}M"
        trainable = f"{r['trainable_params']/1e6:.2f}M"
        ratio = f"{r['trainable_ratio']:.1f}%"
        lora = f"{r['lora_params']/1e6:.2f}M" if r['lora_params'] > 0 else "-"
        flops = f"{r['flops']/1e9:.2f}G" if r['flops'] else "N/A"
        speed = f"{r['inference_time_ms']:.2f}" if r['inference_time_ms'] else "N/A"
        fps = f"{r['fps']:.2f}" if r['fps'] else "N/A"
        
        print(f"{name:<30} {total:<15} {trainable:<15} {ratio:<10} {lora:<12} {flops:<12} {speed:<12} {fps:<10}")
    
    print(f"{'-'*100}")
    
    # Print comparison
    if len(results) > 1:
        baseline = results[0]
        print(f"\n{'='*100}")
        print(f"📈 COMPARISON TO BASELINE")
        print(f"{'='*100}")
        
        print(f"\n{'Model':<30} {'Params Increase':<20} {'FLOPs Increase':<20} {'Speed Change':<20}")
        print(f"{'-'*100}")
        
        for r in results[1:]:
            name = r['name']
            
            # Params increase
            params_inc = (r['total_params'] - baseline['total_params']) / baseline['total_params'] * 100
            params_str = f"+{params_inc:.2f}%" if params_inc > 0 else f"{params_inc:.2f}%"
            
            # FLOPs increase
            if r['flops'] and baseline['flops']:
                flops_inc = (r['flops'] - baseline['flops']) / baseline['flops'] * 100
                flops_str = f"+{flops_inc:.2f}%" if flops_inc > 0 else f"{flops_inc:.2f}%"
            else:
                flops_str = "N/A"
            
            # Speed change
            if r['inference_time_ms'] and baseline['inference_time_ms']:
                speed_inc = (r['inference_time_ms'] - baseline['inference_time_ms']) / baseline['inference_time_ms'] * 100
                speed_str = f"+{speed_inc:.2f}%" if speed_inc > 0 else f"{speed_inc:.2f}%"
            else:
                speed_str = "N/A"
            
            print(f"{name:<30} {params_str:<20} {flops_str:<20} {speed_str:<20}")
        
        print(f"{'-'*100}")


if __name__ == '__main__':
    main()

