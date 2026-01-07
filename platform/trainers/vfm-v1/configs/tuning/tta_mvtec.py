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
text_model_name = 'openai/clip-vit-base-patch32'

# TTA settings
tta_epochs = 10  # TTA iterations (epochs)
tta_lr = 1e-4  # Learning rate for TTA
train_batch_size_per_gpu = 2

# Load baseline model
load_from = 'work_dirs/vfm_v1_l_mvtec/baseline_335_501.pth'

persistent_workers = False

# Model settings - Use YOLOWorldDetector (same as baseline) for TTA
# Freeze backbone and only fine-tune neck/head
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
        frozen_stages=4,  # Freeze image backbone
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),  # Freeze text encoder
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

# Dataset settings - TTA uses prompt_annotations for training
# Text transform for YOLOWorldDetector
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, mask2bbox=True),
    dict(scale=(640, 640), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(640, 640),
        type='LetterResize'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(keys=['gt_masks'], type='RemoveDataElement'),
    *text_transform
]

# TTA training dataset - prompt_annotations (few-shot examples)
tta_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=metainfo_classes,
        data_root='data/mvtec_v2/',
        ann_file='prompt_annotations/annotations.json',
        data_prefix=dict(img='prompt_annotations/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/mvtec.json',
    pipeline=train_pipeline)

train_dataloader = dict(
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=tta_train_dataset)

# Validation dataset - same as original val_annotations
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
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

# Training settings for TTA
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=tta_epochs),
    checkpoint=dict(
        max_keep_ckpts=1,
        save_best='auto',
        interval=tta_epochs))

train_cfg = dict(
    max_epochs=tta_epochs,
    val_interval=1)

# Optimizer - fine-tune neck and head, freeze backbone
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=tta_lr,
        weight_decay=0.0,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.0),  # Freeze backbone
            'neck': dict(lr_mult=1.0),  # Fine-tune neck
            'bbox_head': dict(lr_mult=1.0)  # Fine-tune head
        }),
    constructor='YOLOWv5OptimizerConstructor')

# Evaluation settings
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data/mvtec_v2/val_annotations/annotations.json',
    metric='bbox',
    classwise=True)

test_evaluator = val_evaluator
find_unused_parameters = True

