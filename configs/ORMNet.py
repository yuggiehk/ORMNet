norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
find_unused_parameters=True
dataset_type = 'SeperateObjectEgohos'
data_root='/mnt/nvme1/suyuejiao/egohos_split_data/'
class_hand=3
class_left_obj=2
class_right_obj=2
class_cb=2
crop_size = (448,448)
max_iters=180000

data_preprocessor = dict(
    type='SeperateTwoObjDataPreProcessor',
    mean=[106.01075, 95.40013, 87.42854],
    std=[64.356636, 60.888744, 61.41911],
    bgr_to_rgb=True,# default:bgr
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
    
model = dict(
    type='WithSeperateHeadsforObjCrossAttnSegmentor',
    data_preprocessor=data_preprocessor,
    pretrained='/home/suyuejiao/new_mmseg/mmsegmentation/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth',
    feature_cb_and_hand_to_obj=False,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    decode_head1=dict(
        type='UnetdecoderSeperateHeadsOutputFeature',
        type_decode='hand',
        img_size=crop_size,
        patch_size=4, 
        window_size=7,
        num_classes=class_hand,
        channels=512,
        embed_dim=128,
        depths=[2,2,18,2], 
        num_heads=[4,8,16,32],
        in_chans=3,
        mlp_ratio=4,
        align_corners=False,
        loss_decode=[
                     dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     loss_weight=0.5)],
        ),
    decode_head2=dict(
        type='UnetdecoderSeperateHeadsInputFeatureCrossAttn',
        type_decode='left_obj',
        img_size=crop_size,
        patch_size=4, 
        window_size=7,
        num_classes=class_left_obj,
        channels=512,
        embed_dim=128,
        depths=[2,2,18,2], 
        num_heads=[4,8,16,32],
        in_chans=3,
        mlp_ratio=4,
        align_corners=False,
        loss_decode=[
                     dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     loss_weight=0.5)],
        ),
    decode_head3=dict(
        type='UnetdecoderSeperateHeads',
        type_decode='cb',
        img_size=crop_size,
        patch_size=4, 
        window_size=7,
        num_classes=class_cb,
        channels=512,
        embed_dim=128,
        depths=[2,2,18,2], 
        num_heads=[4,8,16,32],
        in_chans=3,
        mlp_ratio=4,
        align_corners=False,
        loss_decode=[dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     loss_weight=1.0)],
        ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    decode_head4=dict(
        type='UnetdecoderSeperateHeadsInputFeatureCrossAttn',
        type_decode='right_obj',
        img_size=crop_size,
        patch_size=4, 
        window_size=7,
        num_classes=class_right_obj,
        channels=512,
        embed_dim=128,
        depths=[2,2,18,2], 
        num_heads=[4,8,16,32],
        in_chans=3,
        mlp_ratio=4,
        align_corners=False,
        loss_decode=[dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     loss_weight=0.5)],
        ))
    
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/image', seg_map_path='train/label', seg_map_path_hand='train/label_hand',seg_map_path_left_obj='train/lbl_obj_left', seg_map_path_right_obj='train/lbl_obj_right',seg_map_path_two_obj='train/lbl_obj_two', seg_map_path_cb='train/label_contact_first'),
        pipeline=[
            dict(type='LoadMultiLabelImageFromFile'),
            dict(type='LoadSeperateTwoObjAnnotation'),
            dict(
                type='LabelResizeSeperateTwoObj',
                scale=(720, 960),
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(type='RandomSeperateObjectCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSeperateTwoObjLabelSegInputs')
        ]))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=1250)
        
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test_indomain/image', seg_map_path='test_indomain/label', seg_map_path_hand='test_indomain/label_hand',seg_map_path_left_obj='test_indomain/lbl_obj_left', seg_map_path_right_obj='test_indomain/lbl_obj_right',seg_map_path_two_obj='test_indomain/lbl_obj_two', seg_map_path_cb='test_indomain/label_contact_first'),
        pipeline=[
            dict(type='LoadMultiLabelImageFromFile'),
            dict(type='LoadSeperateTwoObjAnnotation'),
            dict(type='ThreeLabelResizeSeperateTwoobj', scale=crop_size, keep_ratio=False),
            dict(type='PackSeperateTwoObjLabelSegInputs')
        ]))
val_evaluator = dict(type='NewSeperateIou', iou_metrics=['mIoU'])
val_cfg = dict(type='ValLoop')

test_dataloader = val_dataloader
test_cfg = val_cfg
test_evaluator = val_evaluator

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

visualizer = dict(
    type='SegLocalVisualizer', vis_backends=[dict(type='TensorboardVisBackend')], name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00001, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=10000),
    dict(
        type='PolyLR',
        eta_min=0,
        power=1.1,
        begin=10000,
        end=max_iters,
        by_epoch=False,
    )
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000, save_optimizer=True, save_param_scheduler=True, save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))