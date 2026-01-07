"""
Baseline evaluation config for MVTec dataset.
Uses the original YOLOWorldDetector for fair comparison.
"""

_base_ = (
    '../../third_party/mmyolo/configs/yolov8/'
    'yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py')

custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False)

metainfo_classes = dict(
    classes=('defect', 'coil', 'discoloration', 'dust')
)

# hyper-parameters
num_classes = 4
num_training_classes = 4
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]

# Load baseline model
load_from = 'work_dirs/vfm_v1_l_mvtec/baseline_335_501.pth'
text_model_name = 'openai/clip-vit-base-patch32'

# Model settings - Original YOLOWorldDetector
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck=dict(
        type='YOLOWorldPAFPN',
        guide_channels=text_channels,
        embed_channels=neck_embed_channels,
        num_heads=neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            use_bn_head=True,
            embed_dims=text_channels,
            num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# Validation dataset
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts'))
]

val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=metainfo_classes,
        data_root='data/mvtec_v2/',
        ann_file='val_annotations/annotations.json',
        data_prefix=dict(img='val_annotations/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/mvtec.json',
    pipeline=test_pipeline)

val_dataloader = dict(dataset=val_dataset)
test_dataloader = val_dataloader

# Evaluation settings
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data/mvtec_v2/val_annotations/annotations.json',
    metric='bbox',
    classwise=True)

test_evaluator = val_evaluator

