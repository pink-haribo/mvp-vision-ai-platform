#!/usr/bin/env python
"""
Simple parameter counting script.

Usage:
    python tools/analysis/simple_param_count.py CONFIG_FILE
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
import yolo_world  # This imports all yolo_world modules

# Register all modules
register_mmyolo(init_default_scope=True)


def measure_inference_speed(model, num_warmup=10, num_iterations=100):
    """Measure inference speed for batch_size=1."""
    import time
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Create dummy input (batch_size=1, 3 channels, 640x640)
    dummy_img = torch.randn(1, 3, 640, 640).to(device)

    # Create dummy text input for YOLO-World
    dummy_texts = [['defect', 'coil', 'discoloration', 'dust']]  # 4 classes

    # Create dummy data sample
    data_sample = DetDataSample()
    data_sample.texts = dummy_texts[0]
    batch_data_samples = [data_sample]

    # Warmup
    print(f"  Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            try:
                _ = model.extract_feat(dummy_img, batch_data_samples)
            except:
                # Fallback: just measure backbone
                _ = model.backbone(dummy_img, dummy_texts)

    # Measure
    print(f"  Measuring speed ({num_iterations} iterations)...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            try:
                _ = model.extract_feat(dummy_img, batch_data_samples)
            except:
                # Fallback: just measure backbone
                _ = model.backbone(dummy_img, dummy_texts)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)

    avg_time = sum(times) / len(times)
    return avg_time * 1000  # Convert to ms


def count_params(config_path):
    """Count parameters for a config file."""
    print(f"\n{'='*80}")
    print(f"Config: {config_path}")
    print(f"{'='*80}\n")

    try:
        # Set default scope
        from mmengine.registry import DefaultScope
        DefaultScope.get_instance('mmyolo', scope_name='mmyolo')

        # Load config
        cfg = Config.fromfile(config_path)

        # Build model
        print("Building model...")
        model = MODELS.build(cfg.model)
        model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # Count LoRA parameters
        lora_params = sum(p.numel() for n, p in model.named_parameters()
                         if 'lora' in n.lower() and p.requires_grad)

        print(f"✅ Model built successfully!\n")
        print(f"📊 Parameters:")
        print(f"  Total:      {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable:  {trainable_params:,} ({trainable_params/1e6:.2f}M) [{trainable_params/total_params*100:.2f}%]")
        print(f"  Frozen:     {frozen_params:,} ({frozen_params/1e6:.2f}M)")

        if lora_params > 0:
            print(f"  LoRA:       {lora_params:,} ({lora_params/1e6:.2f}M) [{lora_params/trainable_params*100:.1f}% of trainable]")

        # Measure inference speed
        print(f"\n⏱️  Measuring inference speed (batch_size=1)...")
        try:
            inference_time = measure_inference_speed(model, num_warmup=10, num_iterations=100)
            print(f"✅ Inference time: {inference_time:.2f} ms/image")
            print(f"  Throughput:     {1000/inference_time:.1f} FPS")
        except Exception as e:
            print(f"❌ Failed to measure speed: {e}")
            inference_time = None

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'lora': lora_params,
            'inference_time': inference_time
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        count_params(config_path)
    else:
        # Test all configs
        configs = [
            'configs/finetune_coco/vfm_v1_l_mvtec.py',
            'configs/adapter/phase2_dense_lora_v1.py',
            'configs/adapter/phase2_hybrid_v1.py',
            'configs/adapter/phase2_hybrid_aggressive_v1.py',
        ]
        
        results = []
        for cfg in configs:
            result = count_params(cfg)
            if result:
                results.append((cfg, result))
        
        # Print summary
        print(f"\n\n{'='*120}")
        print(f"📊 SUMMARY")
        print(f"{'='*120}\n")
        print(f"{'Config':<45} {'Total':<12} {'Trainable':<12} {'Ratio':<8} {'LoRA':<10} {'Inference (ms)':<15} {'FPS':<8}")
        print(f"{'-'*120}")

        for cfg, r in results:
            name = cfg.split('/')[-1].replace('.py', '')
            total = f"{r['total']/1e6:.2f}M"
            trainable = f"{r['trainable']/1e6:.2f}M"
            ratio = f"{r['trainable']/r['total']*100:.1f}%"
            lora = f"{r['lora']/1e6:.2f}M" if r['lora'] > 0 else "-"
            inf_time = f"{r['inference_time']:.2f}" if r['inference_time'] else "N/A"
            fps = f"{1000/r['inference_time']:.1f}" if r['inference_time'] else "N/A"

            print(f"{name:<45} {total:<12} {trainable:<12} {ratio:<8} {lora:<10} {inf_time:<15} {fps:<8}")

