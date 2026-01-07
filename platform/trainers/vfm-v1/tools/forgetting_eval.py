"""
Catastrophic Forgetting Evaluation Script for YOLO-World TTA

This script evaluates whether TTA causes catastrophic forgetting by comparing:
1. Baseline checkpoint performance on train/val data
2. TTA checkpoint performance on train/val data

Usage:
    python tools/forgetting_eval.py \
        --config configs/tuning/tta_mvtec.py \
        --baseline-ckpt work_dirs/vfm_v1_l_mvtec/baseline_335_501.pth \
        --tta-ckpt work_dirs/tta_eval/tta/epoch_20.pth \
        --train-ann data/mvtec_v2/train_annotations/annotations.json \
        --val-ann data/mvtec_v2/val_annotations/annotations.json
"""

import argparse
import os
import os.path as osp
import json
import copy

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description='Catastrophic Forgetting Evaluation for YOLO-World TTA')
    parser.add_argument('--config', 
                        default='configs/tuning/tta_mvtec.py',
                        help='Config file path')
    parser.add_argument('--baseline-ckpt',
                        default='work_dirs/vfm_v1_l_mvtec/baseline_335_501.pth',
                        help='Baseline checkpoint path')
    parser.add_argument('--tta-ckpt',
                        default='work_dirs/tta_eval/tta/epoch_20.pth',
                        help='TTA checkpoint path')
    parser.add_argument('--train-ann',
                        default='data/mvtec_v2/train_annotations/annotations.json',
                        help='Train annotations path')
    parser.add_argument('--val-ann',
                        default='data/mvtec_v2/val_annotations/annotations.json',
                        help='Val annotations path')
    parser.add_argument('--work-dir',
                        default='work_dirs/forgetting_eval',
                        help='Work directory')
    parser.add_argument('--device', default='cuda:0', help='Device')
    args = parser.parse_args()
    return args


def evaluate_checkpoint(cfg, checkpoint, ann_file, img_prefix, work_dir, eval_name):
    """Evaluate a checkpoint on specified dataset."""
    print(f"\n{'='*50}")
    print(f"=== Evaluating: {eval_name} ===")
    print(f"{'='*50}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Annotation: {ann_file}")
    print(f"Image prefix: {img_prefix}")

    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.work_dir = work_dir
    eval_cfg.load_from = checkpoint

    # Update test dataset - use relative paths from data_root
    # data_root is 'data/mvtec_v2/', so ann_file should be relative
    ann_basename = osp.basename(osp.dirname(ann_file))  # e.g., 'train_annotations'
    eval_cfg.test_dataloader.dataset.dataset.ann_file = f'{ann_basename}/annotations.json'
    eval_cfg.test_dataloader.dataset.dataset.data_prefix = dict(img=f'{ann_basename}/')
    eval_cfg.test_evaluator.ann_file = ann_file  # Full path for evaluator

    # Disable training - all three must be None together
    eval_cfg.train_cfg = None
    eval_cfg.train_dataloader = None
    eval_cfg.optim_wrapper = None

    runner = Runner.from_cfg(eval_cfg)
    metrics = runner.test()

    return metrics


def print_results(results):
    """Print formatted results table."""
    print("\n" + "="*70)
    print("=== Catastrophic Forgetting Analysis ===")
    print("="*70)
    
    # Extract metrics
    baseline_train = results['baseline_train']
    tta_train = results['tta_train']
    baseline_val = results['baseline_val']
    tta_val = results['tta_val']
    
    # Key metrics to display
    metrics = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75']
    metric_names = ['mAP', 'mAP_50', 'mAP_75']
    
    print(f"\n{'Metric':<12} {'Train (Baseline)':<18} {'Train (TTA)':<18} {'Δ Train':<12}")
    print("-" * 60)
    
    for metric, name in zip(metrics, metric_names):
        b_train = baseline_train.get(metric, 0)
        t_train = tta_train.get(metric, 0)
        diff = t_train - b_train
        sign = '+' if diff >= 0 else ''
        print(f"{name:<12} {b_train:<18.4f} {t_train:<18.4f} {sign}{diff:<12.4f}")
    
    print(f"\n{'Metric':<12} {'Val (Baseline)':<18} {'Val (TTA)':<18} {'Δ Val':<12}")
    print("-" * 60)
    
    for metric, name in zip(metrics, metric_names):
        b_val = baseline_val.get(metric, 0)
        t_val = tta_val.get(metric, 0)
        diff = t_val - b_val
        sign = '+' if diff >= 0 else ''
        print(f"{name:<12} {b_val:<18.4f} {t_val:<18.4f} {sign}{diff:<12.4f}")
    
    # Summary
    train_forgetting = tta_train.get('coco/bbox_mAP_50', 0) - baseline_train.get('coco/bbox_mAP_50', 0)
    val_gain = tta_val.get('coco/bbox_mAP_50', 0) - baseline_val.get('coco/bbox_mAP_50', 0)
    
    print("\n" + "="*70)
    print("=== Summary (mAP_50) ===")
    print("="*70)
    print(f"Train Forgetting: {train_forgetting:+.4f} ({train_forgetting*100:+.2f}%)")
    print(f"Val Gain:         {val_gain:+.4f} ({val_gain*100:+.2f}%)")
    
    if train_forgetting < -0.05:
        conclusion = "⚠️  Significant forgetting detected!"
    elif train_forgetting < -0.02:
        conclusion = "⚡ Mild forgetting, acceptable trade-off"
    else:
        conclusion = "✅ No significant forgetting"
    
    if val_gain > 0.05:
        conclusion += " | Strong generalization gain"
    elif val_gain > 0:
        conclusion += " | Positive generalization"
    else:
        conclusion += " | No generalization improvement"
    
    print(f"\nConclusion: {conclusion}")

    return {
        'train_forgetting': train_forgetting,
        'val_gain': val_gain
    }


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    os.makedirs(args.work_dir, exist_ok=True)

    results = {}

    # 1. Baseline on Train
    results['baseline_train'] = evaluate_checkpoint(
        cfg, args.baseline_ckpt, args.train_ann, 'train_annotations/',
        osp.join(args.work_dir, 'baseline_train'), 'Baseline on Train')

    # 2. Baseline on Val
    results['baseline_val'] = evaluate_checkpoint(
        cfg, args.baseline_ckpt, args.val_ann, 'val_annotations/',
        osp.join(args.work_dir, 'baseline_val'), 'Baseline on Val')

    # 3. TTA on Train
    results['tta_train'] = evaluate_checkpoint(
        cfg, args.tta_ckpt, args.train_ann, 'train_annotations/',
        osp.join(args.work_dir, 'tta_train'), 'TTA on Train')

    # 4. TTA on Val
    results['tta_val'] = evaluate_checkpoint(
        cfg, args.tta_ckpt, args.val_ann, 'val_annotations/',
        osp.join(args.work_dir, 'tta_val'), 'TTA on Val')

    # Print and analyze results
    summary = print_results(results)

    # Save results
    output = {
        'baseline_train': results['baseline_train'],
        'baseline_val': results['baseline_val'],
        'tta_train': results['tta_train'],
        'tta_val': results['tta_val'],
        'summary': summary,
        'config': {
            'baseline_ckpt': args.baseline_ckpt,
            'tta_ckpt': args.tta_ckpt,
            'train_ann': args.train_ann,
            'val_ann': args.val_ann
        }
    }

    output_path = osp.join(args.work_dir, 'forgetting_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

