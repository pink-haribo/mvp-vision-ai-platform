"""
Test-Time Adaptation (TTA) Evaluation Script for YOLO-World

This script performs:
1. Baseline evaluation (without TTA)
2. Image prompt-based embedding initialization
3. Prompt tuning adaptation
4. Post-TTA evaluation

Usage:
    python tools/tta_eval.py configs/tuning/tta_mvtec.py \
        --checkpoint work_dirs/vfm_v1_l_mvtec/baseline_335_501.pth \
        --tta-epochs 10 --tta-lr 1e-3
"""

import argparse
import os
import os.path as osp
import json
import copy
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.registry import MODELS
from mmdet.apis import init_detector
from mmdet.evaluation import CocoMetric

from transformers import CLIPVisionModelWithProjection, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description='TTA Evaluation for YOLO-World')
    parser.add_argument('config', help='TTA config file path')
    parser.add_argument('--baseline-config',
                        default='configs/tuning/baseline_eval_mvtec.py',
                        help='Baseline config file path')
    parser.add_argument('--checkpoint', help='Checkpoint file path')
    parser.add_argument('--tta-epochs', type=int, default=10, help='TTA epochs')
    parser.add_argument('--tta-lr', type=float, default=1e-3, help='TTA learning rate')
    parser.add_argument('--work-dir', default='work_dirs/tta_eval', help='Work directory')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline evaluation')
    parser.add_argument('--use-image-prompt', action='store_true',
                        help='Initialize embeddings from image prompts')
    args = parser.parse_args()
    return args


def generate_image_embeddings(prompt_ann_path, prompt_img_dir, categories, device='cuda:0'):
    """Generate embeddings from prompt images using CLIP vision encoder."""
    print("\n=== Generating Image Prompt Embeddings ===")
    
    # Load CLIP vision model
    vision_model = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch32')
    processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
    vision_model.to(device)
    vision_model.eval()
    
    # Load annotations
    with open(prompt_ann_path, 'r') as f:
        ann_data = json.load(f)
    
    # Category id to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in ann_data['categories']}
    cat_name_to_idx = {name: idx for idx, name in enumerate(categories)}
    
    # Image id to file mapping
    img_id_to_file = {img['id']: img['file_name'] for img in ann_data['images']}
    
    # Collect embeddings per category
    cat_embeddings = {cat: [] for cat in categories}
    
    for ann in ann_data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        bbox = ann['bbox']  # [x, y, w, h]
        
        cat_name = cat_id_to_name[cat_id]
        if cat_name not in cat_name_to_idx:
            continue
            
        # Load and crop image
        img_path = osp.join(prompt_img_dir, img_id_to_file[img_id])
        image = Image.open(img_path).convert('RGB')
        
        # Crop region (with some padding)
        x, y, w, h = bbox
        pad = 0.1
        x1 = max(0, int(x - w * pad))
        y1 = max(0, int(y - h * pad))
        x2 = min(image.width, int(x + w * (1 + pad)))
        y2 = min(image.height, int(y + h * (1 + pad)))
        
        cropped = image.crop((x1, y1, x2, y2))
        
        # Get CLIP embedding
        inputs = processor(images=cropped, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = vision_model(**inputs)
            img_feat = outputs.image_embeds
            img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)
        
        cat_embeddings[cat_name].append(img_feat.cpu())
    
    # Average embeddings per category
    final_embeddings = []
    for cat in categories:
        if cat_embeddings[cat]:
            avg_emb = torch.cat(cat_embeddings[cat], dim=0).mean(dim=0, keepdim=True)
            avg_emb = avg_emb / avg_emb.norm(p=2, dim=-1, keepdim=True)
            final_embeddings.append(avg_emb)
            print(f"  {cat}: {len(cat_embeddings[cat])} samples")
        else:
            # Use text embedding as fallback
            print(f"  {cat}: No samples, using text embedding")
            text_emb = np.load('embeddings/mvtec_4_embeddings.npy')
            idx = categories.index(cat)
            final_embeddings.append(torch.from_numpy(text_emb[idx:idx+1]))
    
    embeddings = torch.cat(final_embeddings, dim=0)
    print(f"Final embeddings shape: {embeddings.shape}")
    return embeddings


def run_baseline_eval(baseline_config_path, checkpoint, work_dir, device):
    """Run baseline evaluation without TTA."""
    print("\n" + "="*50)
    print("=== Baseline Evaluation (No TTA) ===")
    print("="*50)

    # Load baseline config (uses YOLOWorldDetector)
    baseline_cfg = Config.fromfile(baseline_config_path)
    baseline_cfg.work_dir = osp.join(work_dir, 'baseline')
    baseline_cfg.load_from = checkpoint

    runner = Runner.from_cfg(baseline_cfg)
    metrics = runner.test()

    return metrics


def run_tta_training(cfg, checkpoint, init_embeddings, tta_epochs, tta_lr, device):
    """Run TTA training with prompt tuning."""
    print("\n" + "="*50)
    print("=== Test-Time Adaptation Training ===")
    print("="*50)

    tta_cfg = copy.deepcopy(cfg)
    tta_cfg.work_dir = osp.join(cfg.work_dir, 'tta')
    tta_cfg.load_from = checkpoint

    # Update TTA settings
    tta_cfg.train_cfg.max_epochs = tta_epochs
    tta_cfg.optim_wrapper.optimizer.lr = tta_lr
    tta_cfg.default_hooks.param_scheduler.max_epochs = tta_epochs

    # Save initial embeddings if provided (only for SimpleYOLOWorldDetector)
    if init_embeddings is not None and tta_cfg.model.type == 'SimpleYOLOWorldDetector':
        init_emb_path = osp.join(tta_cfg.work_dir, 'init_embeddings.npy')
        os.makedirs(tta_cfg.work_dir, exist_ok=True)
        np.save(init_emb_path, init_embeddings.numpy())
        tta_cfg.model.embedding_path = init_emb_path
        print(f"Using image prompt embeddings: {init_emb_path}")

    runner = Runner.from_cfg(tta_cfg)
    runner.train()

    return runner


def run_tta_eval(runner):
    """Run evaluation after TTA."""
    print("\n" + "="*50)
    print("=== Post-TTA Evaluation ===")
    print("="*50)

    metrics = runner.test()
    return metrics


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)

    # Get categories
    categories = list(cfg.metainfo_classes.classes)
    print(f"Categories: {categories}")

    checkpoint = args.checkpoint or cfg.load_from

    # 1. Baseline evaluation
    baseline_metrics = None
    if not args.skip_baseline:
        baseline_metrics = run_baseline_eval(
            args.baseline_config, checkpoint, args.work_dir, args.device)
        print("\n=== Baseline Results ===")
        for k, v in baseline_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 2. Generate image prompt embeddings (optional)
    init_embeddings = None
    if args.use_image_prompt:
        prompt_ann_path = 'data/mvtec_v2/prompt_annotations/annotations.json'
        prompt_img_dir = 'data/mvtec_v2/prompt_annotations'
        init_embeddings = generate_image_embeddings(
            prompt_ann_path, prompt_img_dir, categories, args.device)

    # 3. TTA training
    runner = run_tta_training(
        cfg, checkpoint, init_embeddings,
        args.tta_epochs, args.tta_lr, args.device)

    # 4. Post-TTA evaluation
    tta_metrics = run_tta_eval(runner)

    # 5. Print comparison
    print("\n" + "="*50)
    print("=== Results Comparison ===")
    print("="*50)

    if baseline_metrics:
        print("\nBaseline:")
        for k, v in baseline_metrics.items():
            if 'mAP' in k or 'AP' in k:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nAfter TTA:")
    for k, v in tta_metrics.items():
        if 'mAP' in k or 'AP' in k:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save results
    results = {
        'baseline': baseline_metrics,
        'tta': tta_metrics,
        'config': {
            'tta_epochs': args.tta_epochs,
            'tta_lr': args.tta_lr,
            'use_image_prompt': args.use_image_prompt
        }
    }

    results_path = osp.join(args.work_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()

