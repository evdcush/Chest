## https://github.com/SwinTransformer/AiT

## Train VAE FIRST! Then "task solver".

#=============================================================================#
#                                                                             #
#                           ██████ ███████  ██████                            #
#                          ██      ██      ██                                 #
#                          ██      █████   ██   ███                           #
#                          ██      ██      ██    ██                           #
#                           ██████ ██       ██████                            #
#                                                                             #
#=============================================================================#
##  ait/configs/swinv2b_640reso_joint.py

checkpoint_config = dict(interval=12115)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]

num_bins = 2000
num_classes = 80
num_embeddings = 128
num_embeddings_depth = 128
num_embeddings_others = 128  # other tasks token
num_vocal = num_bins+1 + num_classes + 2 + num_embeddings + \
    num_embeddings_depth + num_embeddings_others

model = dict(
    type='AiT',
    padto=640,
    backbone=dict(
        type="SwinV2TransformerRPE2FC",
        pretrain_img_size=192,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[30, 30, 30, 15],
        use_shift=[True, True, True, True],
        pretrain_window_size=[12, 12, 12, 6],
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        out_indices=(3, ),
        init_cfg=dict(type='Pretrained',
                      checkpoint='swin_v2_base_densesimmim.pth'),
    ),
    transformer=dict(type='ARTransformer',
                     in_chans=1024,
                     d_model=256,
                     drop_path=0.1,
                     drop_out=0.1,
                     nhead=8,
                     dim_feedforward=1024,
                     num_encoder_layers=6,
                     num_decoder_layers=6,
                     num_vocal=num_vocal,
                     num_bins=num_bins,
                     num_classes=num_classes,
                     num_embeddings=num_embeddings,
                     num_embeddings_depth=num_embeddings_depth,
                     dec_length=2100,
                     n_rows=20,
                     n_cols=20,
                     pos_enc='sine',
                     pred_eos=False,
                     soft_vae=True,
                     soft_transformer=True,
                     top_p=0.3),
    task_heads=dict(
        insseg=dict(
            type='InsSegHead',
            task_id=1,
            loss_weight=1.,
            num_classes=num_classes,
            num_bins=num_bins,
            coord_norm='abs',  # abs or rel
            norm_val=640,
            sync_cls_avg_factor=True,
            vae_cfg=dict(type='VQVAE',
                         token_length=16,
                         mask_size=64,
                         embedding_dim=512,
                         hidden_dim=128,
                         num_resnet_blocks=2,
                         num_embeddings=num_embeddings,
                         pretrained='vqvae_insseg.pt',
                         freeze=True),
            mask_weight=0.2,
            decoder_loss_weight=5.0,
            max_obj_decoderloss=100,
            seq_aug=True),
        depth=dict(type='DepthHead',
                   task_id=2,
                   loss_weight=0.2,
                   depth_token_offset=num_bins + 1 + num_classes + 2 +
                   num_embeddings,
                   vae_cfg=dict(type='VQVAE',
                                use_norm=False,
                                token_length=15 * 15,
                                mask_size=480,
                                embedding_dim=512,
                                hidden_dim=256,
                                num_resnet_blocks=2,
                                num_embeddings=num_embeddings_depth,
                                pretrained='vqvae_depth.pt',
                                freeze=True),
                   decoder_loss_weight=1.0,
                   soft_vae=True),
    ))

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

# large scale jitter pipeline from configs/common/lsj_100e_coco_instance.py
image_size = (640, 640)
file_client_args = dict(backend='disk')

runner = dict(type='IterBasedRunnerMultitask', max_iters=12115 * 25)
evaluation = dict(interval=12115 * 25)

# learning policy

lr_config = dict(
    policy='LinearAnnealing',
    by_epoch=False,
    min_lr_ratio=0.01,
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=12115,
)

# optimizer
optimizer = dict(type='AdamW',
                 lr=8e-4,
                 weight_decay=0.05,
                 constructor='SwinLayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(
                     num_layers=[2, 2, 18, 2],
                     layer_decay_rate=0.85,
                     no_decay_names=[
                         'relative_position_bias_table', 'rpe_mlp',
                         'logit_scale', 'det_embed', 'voc_embed', 'enc_embed',
                         'dec_embed', 'mask_embed'
                     ],
                 ))
optimizer_config = dict(grad_clip={'max_norm': 10, 'norm_type': 2})

insseg_train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=image_size,
         ratio_range=(0.1, 3.0),
         multiscale_mode='range',
         keep_ratio=True),
    dict(type='RandomCrop',
         crop_type='absolute',
         crop_size=image_size,
         recompute_bbox=False,
         allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    dict(type='AddKey', kv={'task_type': 'insseg'})
]

insseg_test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='MultiScaleFlipAug',
         img_scale=(640, 640),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size=image_size),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]

det_dataset_type = 'CocoDataset'

task = dict(
    insseg=dict(  # len=117266
        times=1,
        data=dict(
            train=dict(
                type=det_dataset_type,
                ann_file='data/coco/annotations/instances_train2017.json',
                img_prefix='data/coco/train2017/',
                pipeline=insseg_train_pipeline,
                samples_total_gpu=60),
            val=dict(type=det_dataset_type,
                     ann_file='data/coco/annotations/instances_val2017.json',
                     img_prefix='data/coco/val2017/',
                     pipeline=insseg_test_pipeline,
                     samples_per_gpu=8,
                     workers_per_gpu=4),
        )),
    depth=dict(  # len=24231
        times=1,
        data=dict(
            train=dict(type='nyudepthv2',
                       data_path='data',
                       filenames_path='code/dataset/depth/filenames/',
                       is_train=True,
                       crop_size=(480, 480),
                       samples_total_gpu=4),
            val=dict(type='nyudepthv2',
                     data_path='data',
                     filenames_path='code/dataset/depth/filenames/',
                     is_train=False,
                     crop_size=(480, 480),
                     samples_per_gpu=2,
                     workers_per_gpu=8),
        )))

# enable fp16
fp16 = dict(loss_scale='dynamic')

load_from = 'ait_det_swinv2b_wodec.pth'

#█████████████████████████████████████████████████████████████████████████████#
#                                    VQ-VAE                                   #
#█████████████████████████████████████████████████████████████████████████████#

#=============================================================================#
#                     vae/configs/depth/ait_depth_vqvae.py                    #
#=============================================================================#

image_size = 480
model = dict(
    image_size=image_size,
    num_resnet_blocks=2,
    downsample_ratio=32,
    num_tokens=128,
    codebook_dim=512,
    hidden_dim=16,
    use_norm=False,
    channels=1,
    train_objective='regression',
    max_value=10.,
    residul_type='v1',
    loss_type='mse_ignore_zero',
)

train_setting = dict(output_dir='outputs',
                     data=dict(
                         is_train=True,
                         data_path='data/nyu_depth_v2',
                         filenames_path='./dataset/filenames',
                         mask=True,
                         mask_ratio=0.5,
                         mask_patch_size=16,
                         crop_size=(image_size, image_size),
                     ),
                     opt_params=dict(
                         epochs=20,
                         batch_size=8,
                         learning_rate=3e-4,
                         lr_decay_rate=0.98,
                         schedule_step=500,
                         schedule_type='exp',
                     ))

test_setting = dict(data=dict(
    data_path='data/nyu_depth_v2',
    filenames_path='./dataset/filenames',
), )

#=============================================================================#
#                    vae/configs/insseg/ait_insseg_vqvae.py                   #
#=============================================================================#

from torchvision import transforms as T

image_size = 64
model = dict(
    image_size=image_size,
    num_resnet_blocks=2,
    downsample_ratio=16,
    num_tokens=128,
    codebook_dim=512,
    hidden_dim=16,
    channels=1,
    use_norm=True,
    train_objective='regression',
    max_value=1.,
    residul_type='v1',
    loss_type='mse',
)

train_setting = dict(output_dir='outputs/',
                     data=dict(
                         image_folder='data/maskcoco/instances_train2017',
                         pipeline=[
                             dict(type='Resize',
                                  size=image_size,
                                  interpolation=T.InterpolationMode.BILINEAR),
                             dict(type='CenterCrop', size=image_size),
                             dict(type='CustomToTensor'),
                             dict(type='Uint8Remap'),
                         ],
                     ),
                     opt_params=dict(
                         epochs=20,
                         batch_size=512,
                         learning_rate=3e-4,
                         warmup_ratio=1e-3,
                         warmup_steps=500,
                         weight_decay=0.0,
                         schedule_type='cosine',
                     ))

test_setting = dict(
    coco_dir='data/coco',
    target_size=(image_size, image_size),
    iou_type=['segm', 'boundary'],
    max_samples=5000,
    seed=1234,
)

#=============================================================================#
#                                                                             #
#            ██████  ██    ██ ███    ██ ███    ██ ███████ ██████              #
#            ██   ██ ██    ██ ████   ██ ████   ██ ██      ██   ██             #
#            ██████  ██    ██ ██ ██  ██ ██ ██  ██ █████   ██████              #
#            ██   ██ ██    ██ ██  ██ ██ ██  ██ ██ ██      ██   ██             #
#            ██   ██  ██████  ██   ████ ██   ████ ███████ ██   ██             #
#                                                                             #
#=============================================================================#
## ait/code/runner/iter_based_runner_multitask.py

import os.path as osp
import platform
import shutil
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, no_type_check

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mmcv
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.hooks import IterTimerHook
from mmcv.runner.utils import get_host_info


class IterLoader:

    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader
        print('dataloader length (iters/epoch):', len(dataloader))
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


@RUNNERS.register_module()
class IterBasedRunnerMultitask(BaseRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        imgs, gts, tasks = [], [], []
        gt = next(data_loader)
        imgs, gts, tasks = [], [], []
        for task, v in gt.items():
            img = v.pop('img')
            imgs.append(img)
            gts.append(v)
            tasks.append(task)

        data_batch = {'img': imgs, 'gt': gts, 'task': tasks}
        self.call_hook('before_train_iter')
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_val_iter')
        outputs = self.model.val_step(data_batch, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.val_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_val_iter')
        del self.data_batch
        self._inner_iter += 1

    def run(self,
            data_loaders: DataLoader,
            max_iters: Optional[int] = None,
            **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        # assert isinstance(data_loaders, list)
        # assert len(data_loaders) == len(task_times)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        # self.logger.info('task_times: %s, max: %d iters', task_times,
        #                  self._max_iters)
        self.call_hook('before_run')

        iter_loaders = iter(data_loaders)

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            # for i, (name, times) in enumerate(task_times):
            self._inner_iter = 0
            self.train(iter_loaders)
            # for _ in range(times):
            #     if self.iter >= self._max_iters:
            #         break
            #     self.train(iter_loaders[i], name, **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

    @no_type_check
    def resume(self,
               checkpoint: str,
               resume_optimizer: bool = True,
               map_location: Union[str, Callable] = 'default') -> None:
        """Resume model from checkpoint.

        Args:
            checkpoint (str): Checkpoint to resume from.
            resume_optimizer (bool, optional): Whether resume the optimizer(s)
                if the checkpoint file includes optimizer(s). Default to True.
            map_location (str, optional): Same as :func:`torch.load`.
                Default to 'default'.
        """
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(checkpoint,
                                              map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        self._inner_iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        self.logger.info(f'resumed from epoch: {self.epoch}, iter {self.iter}')

    def save_checkpoint(  # type: ignore
            self,
            out_dir: str,
            filename_tmpl: str = 'iter_{}.pth',
            meta: Optional[Dict] = None,
            save_optimizer: bool = True,
            create_symlink: bool = True) -> None:
        """Save checkpoint to file.

        Args:
            out_dir (str): Directory to save checkpoint files.
            filename_tmpl (str, optional): Checkpoint file template.
                Defaults to 'iter_{}.pth'.
            meta (dict, optional): Metadata to be saved in checkpoint.
                Defaults to None.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            try:
                mmcv.symlink(filename, dst_file)
            except:
                shutil.copy(filepath, dst_file)

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None,
                                custom_hooks_config=None):
        """Register default hooks for iter-based training.

        Checkpoint hook, optimizer stepper hook and logger hooks will be set to
        `by_epoch=False` by default.

        Default hooks include:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | LrUpdaterHook        | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | MomentumUpdaterHook  | HIGH (30)               |
        +----------------------+-------------------------+
        | OptimizerStepperHook | ABOVE_NORMAL (40)       |
        +----------------------+-------------------------+
        | CheckpointSaverHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | IterTimerHook        | LOW (70)                |
        +----------------------+-------------------------+
        | LoggerHook(s)        | VERY_LOW (90)           |
        +----------------------+-------------------------+
        | CustomHook(s)        | defaults to NORMAL (50) |
        +----------------------+-------------------------+

        If custom hooks have same priority with default hooks, custom hooks
        will be triggered after default hooks.
        """
        if checkpoint_config is not None:
            checkpoint_config.setdefault('by_epoch', False)  # type: ignore
        if lr_config is not None:
            lr_config.setdefault('by_epoch', False)  # type: ignore
        if log_config is not None:
            for info in log_config['hooks']:
                info.setdefault('by_epoch', False)
        super().register_training_hooks(
            lr_config=lr_config,
            momentum_config=momentum_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
            log_config=log_config,
            timer_config=IterTimerHook(),
            custom_hooks_config=custom_hooks_config)


#=============================================================================#
#                                                                             #
#                 ███    ███  ██████  ██████  ███████ ██                      #
#                 ████  ████ ██    ██ ██   ██ ██      ██                      #
#                 ██ ████ ██ ██    ██ ██   ██ █████   ██                      #
#                 ██  ██  ██ ██    ██ ██   ██ ██      ██                      #
#                 ██      ██  ██████  ██████  ███████ ███████                 #
#                                                                             #
#=============================================================================#

#=============================================================================#
#                 ait/code/model/swin_transformer_v2_rpe2fc.py                #
#=============================================================================#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.utils.checkpoint as checkpoint
from torch.nn.utils import weight_norm
from torch import Tensor, Size
from typing import Union, List
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

from timm.models.layers import DropPath, to_2tuple, trunc_normal_, to_ntuple
from mmcv.runner import _load_checkpoint, load_state_dict
from mmdet.utils import get_root_logger
from .utils import MODELS
from scipy import interpolate
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from collections import OrderedDict

_shape_t = Union[int, List[int], Size]


def custom_normalize(input, p=2, dim=1, eps=1e-12, out=None):
    if out is None:
        denom = input.norm(p, dim, keepdim=True).expand_as(input)
        return input / (denom + eps)
    else:
        denom = input.norm(p, dim, keepdim=True).expand_as(input)
        return torch.div(input, denom + eps, out=out)


class LayerNorm2D(nn.Module):

    def __init__(self, normalized_shape, norm_layer=None):
        super().__init__()
        self.ln = norm_layer(
            normalized_shape) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        """
        x: N C H W
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class LayerNormFP32(nn.LayerNorm):

    def __init__(self,
                 normalized_shape: _shape_t,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True) -> None:
        super(LayerNormFP32, self).__init__(normalized_shape, eps,
                                            elementwise_affine)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input.float(), self.normalized_shape,
                            self.weight.float(), self.bias.float(),
                            self.eps).type_as(input)


class LinearFP32(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearFP32, self).__init__(in_features, out_features, bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input.float(), self.weight.float(),
                        self.bias.float() if self.bias is not None else None)


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 norm_layer=None,
                 mlpfp32=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlpfp32 = mlpfp32

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        if norm_layer is not None:
            self.norm = norm_layer(hidden_features)
        else:
            self.norm = None

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.norm:
            x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        if self.mlpfp32:
            x = self.fc2.float()(x.type(torch.float32))
            x = self.drop.float()(x)
            # print(f"======>[MLP FP32]")
        else:
            x = self.fc2(x)
            x = self.drop(x)
        return x


class ConvMlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 norm_layer=None,
                 mlpfp32=False,
                 proj_ln=False):
        super().__init__()
        self.mlp = Mlp(in_features=in_features,
                       hidden_features=hidden_features,
                       out_features=out_features,
                       act_layer=act_layer,
                       drop=drop,
                       norm_layer=norm_layer,
                       mlpfp32=mlpfp32)
        self.conv_proj = nn.Conv2d(in_features,
                                   in_features,
                                   kernel_size=3,
                                   padding=1,
                                   stride=1,
                                   bias=False,
                                   groups=in_features)
        self.proj_ln = LayerNorm2D(in_features,
                                   LayerNormFP32) if proj_ln else None

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B C H W
        x = self.conv_proj(x)
        if self.proj_ln:
            x = self.proj_ln(x)
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = x.reshape(B, L, C)
        x = self.mlp(x, H, W)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size,
               C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 relative_coords_table_type='norm8_log',
                 rpe_hidden_dim=512,
                 rpe_output_type='normal',
                 attn_type='normal',
                 mlpfp32=False,
                 pretrain_window_size=-1,
                 head_chunk_size=1):

        super().__init__()
        self.head_chunk_size = head_chunk_size
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.mlpfp32 = mlpfp32
        self.attn_type = attn_type
        self.rpe_output_type = rpe_output_type
        self.relative_coords_table_type = relative_coords_table_type

        if self.attn_type == 'cosine_mh':
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(
                (num_heads, 1, 1))),
                                            requires_grad=True)
        elif self.attn_type == 'normal':
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim**-0.5
        else:
            raise NotImplementedError()
        if self.relative_coords_table_type != "none":
            # mlp to generate table of relative position bias
            self.rpe_mlp = nn.Sequential(
                nn.Linear(2, rpe_hidden_dim, bias=True), nn.ReLU(inplace=True),
                LinearFP32(rpe_hidden_dim, num_heads, bias=False))

            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size[0] - 1),
                                             self.window_size[0],
                                             dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[1] - 1),
                                             self.window_size[1],
                                             dtype=torch.float32)
            relative_coords_table = torch.stack(
                torch.meshgrid([relative_coords_h, relative_coords_w
                                ])).permute(1, 2, 0).contiguous().unsqueeze(
                                    0)  # 1, 2*Wh-1, 2*Ww-1, 2
            if relative_coords_table_type == 'linear':
                relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            elif relative_coords_table_type == 'linear_bylayer':
                print(
                    f"norm8_log_bylayer: [{self.window_size}] ==> [{pretrain_window_size}]"
                )
                relative_coords_table[:, :, :, 0] /= (pretrain_window_size - 1)
                relative_coords_table[:, :, :, 1] /= (pretrain_window_size - 1)
            elif relative_coords_table_type == 'norm8_log':
                relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(
                    relative_coords_table) * torch.log2(
                        torch.abs(relative_coords_table) + 1.0) / np.log2(
                            8)  # log8
            elif relative_coords_table_type == 'norm8_log_192to640':
                if self.window_size[0] == 40:
                    relative_coords_table[:, :, :, 0] /= (11)
                    relative_coords_table[:, :, :, 1] /= (11)
                elif self.window_size[0] == 20:
                    relative_coords_table[:, :, :, 0] /= (5)
                    relative_coords_table[:, :, :, 1] /= (5)
                else:
                    raise NotImplementedError
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(
                    relative_coords_table) * torch.log2(
                        torch.abs(relative_coords_table) + 1.0) / np.log2(
                            8)  # log8
            # check
            elif relative_coords_table_type == 'norm8_log_256to640':
                if self.window_size[0] == 40:
                    relative_coords_table[:, :, :, 0] /= (15)
                    relative_coords_table[:, :, :, 1] /= (15)
                elif self.window_size[0] == 20:
                    relative_coords_table[:, :, :, 0] /= (7)
                    relative_coords_table[:, :, :, 1] /= (7)
                else:
                    raise NotImplementedError
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(
                    relative_coords_table) * torch.log2(
                        torch.abs(relative_coords_table) + 1.0) / np.log2(
                            8)  # log8
            elif relative_coords_table_type == 'norm8_log_bylayer':
                print(
                    f"norm8_log_bylayer: [{self.window_size}] ==> [{pretrain_window_size}]"
                )
                relative_coords_table[:, :, :, 0] /= (pretrain_window_size - 1)
                relative_coords_table[:, :, :, 1] /= (pretrain_window_size - 1)
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(
                    relative_coords_table) * torch.log2(
                        torch.abs(relative_coords_table) + 1.0) / np.log2(
                            8)  # log8
            else:
                raise NotImplementedError
            self.register_buffer("relative_coords_table",
                                 relative_coords_table)
        else:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                    num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            trunc_normal_(self.relative_position_bias_table, std=.02)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias,
                                               requires_grad=False),
                 self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.head_chunk_size == 1:
            if self.relative_coords_table_type != "none":
                # relative_position_bias_table: 2*Wh-1 * 2*Ww-1, nH
                relative_position_bias_table = self.rpe_mlp(
                    self.relative_coords_table).view(-1, self.num_heads)
            else:
                relative_position_bias_table = self.relative_position_bias_table
            relative_position_bias = relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1],
                    self.window_size[0] * self.window_size[1],
                    -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            if self.attn_type == 'cosine_mh':
                q = custom_normalize(q.float(), dim=-1, eps=5e-5)
                k = custom_normalize(k.float(), dim=-1, eps=5e-5)
                logit_scale = torch.clamp(
                    self.logit_scale,
                    max=torch.log(
                        torch.tensor(1. / 0.01,
                                     device=self.logit_scale.device))).exp()
                attn = (q @ k.transpose(-2, -1)) * logit_scale.float()
            elif self.attn_type == 'normal':
                q = q * self.scale
                attn = (q.float() @ k.float().transpose(-2, -1))
            else:
                raise NotImplementedError()

            if self.rpe_output_type == 'normal':
                pass
            elif self.rpe_output_type == 'sigmoid':
                relative_position_bias = 16 * \
                    torch.sigmoid(relative_position_bias)
            else:
                raise NotImplementedError

            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N,
                                 N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)

            attn = self.softmax(attn)
            attn = attn.type_as(x)
            attn = self.attn_drop(attn)

            x = attn @ v

        else:
            q_split = torch.split(q.float(),
                                  self.num_heads // self.head_chunk_size,
                                  dim=1)
            k_split = torch.split(k.float(),
                                  self.num_heads // self.head_chunk_size,
                                  dim=1)
            v_split = torch.split(v,
                                  self.num_heads // self.head_chunk_size,
                                  dim=1)
            if self.relative_coords_table_type != "none":
                # relative_position_bias_table: 2*Wh-1 * 2*Ww-1, nH
                relative_position_bias_table = self.rpe_mlp(
                    self.relative_coords_table).view(-1, self.num_heads)
            else:
                relative_position_bias_table = self.relative_position_bias_table
            relative_position_bias = relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1],
                    self.window_size[0] * self.window_size[1],
                    -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias_split = torch.split(relative_position_bias,
                                                       self.num_heads //
                                                       self.head_chunk_size,
                                                       dim=0)
            if self.attn_type == 'cosine_mh':
                logit_scale_split = torch.split(self.logit_scale,
                                                self.num_heads //
                                                self.head_chunk_size,
                                                dim=0)
            x_split = []
            for i_c in range(self.head_chunk_size):
                if self.attn_type == 'cosine_mh':
                    q = custom_normalize(q_split[i_c].float(),
                                         dim=-1,
                                         eps=5e-5)
                    k = custom_normalize(k_split[i_c].float(),
                                         dim=-1,
                                         eps=5e-5)
                    logit_scale = torch.clamp(
                        logit_scale_split[i_c],
                        max=torch.log(torch.tensor(1. / 0.01))).exp()
                    attn = (q @ k.transpose(-2, -1)) * logit_scale.float()
                elif self.attn_type == 'normal':
                    q = q_split[i_c] * self.scale
                    attn = (q.float() @ k_split[i_c].float().transpose(-2, -1))
                else:
                    raise NotImplementedError()

                if self.rpe_output_type == 'normal':
                    pass
                elif self.rpe_output_type == 'sigmoid':
                    relative_position_bias = 16 * \
                        torch.sigmoid(relative_position_bias_split[i_c])
                else:
                    raise NotImplementedError

                attn = attn + relative_position_bias.unsqueeze(0)

                if mask is not None:
                    nW = mask.shape[0]
                    attn = attn.view(B_ // nW, nW,
                                     self.num_heads // self.head_chunk_size, N,
                                     N) + mask.unsqueeze(1).unsqueeze(0)
                    attn = attn.view(-1,
                                     self.num_heads // self.head_chunk_size, N,
                                     N)

                attn = self.softmax(attn)
                attn = attn.type_as(x)
                attn = self.attn_drop(attn)

                x = attn @ v_split[i_c]
                x_split.append(x)

            x = torch.cat(x_split, dim=1)

        x = x.transpose(1, 2).reshape(B_, N, C)

        if self.mlpfp32:
            x = self.proj.float()(x.type(torch.float32))
            x = self.proj_drop.float()(x)
            # print(f"======>[ATTN FP32]")
        else:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlockPost(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 use_mlp_norm=False,
                 endnorm=False,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 relative_coords_table_type='norm8_log',
                 rpe_hidden_dim=512,
                 rpe_output_type='normal',
                 attn_type='normal',
                 mlp_type='normal',
                 mlpfp32=False,
                 pretrain_window_size=-1,
                 head_chunk_size=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_mlp_norm = use_mlp_norm
        self.endnorm = endnorm
        self.mlpfp32 = mlpfp32
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            relative_coords_table_type=relative_coords_table_type,
            rpe_output_type=rpe_output_type,
            rpe_hidden_dim=rpe_hidden_dim,
            attn_type=attn_type,
            mlpfp32=mlpfp32,
            pretrain_window_size=pretrain_window_size,
            head_chunk_size=head_chunk_size)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if mlp_type == 'normal':
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                norm_layer=norm_layer if self.use_mlp_norm else None,
                mlpfp32=mlpfp32)
        elif mlp_type == 'conv':
            self.mlp = ConvMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                norm_layer=norm_layer if self.use_mlp_norm else None,
                mlpfp32=mlpfp32)
        elif mlp_type == 'conv_ln':
            self.mlp = ConvMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                norm_layer=norm_layer if self.use_mlp_norm else None,
                mlpfp32=mlpfp32,
                proj_ln=True)

        if self.endnorm:
            self.enorm = norm_layer(dim)
        else:
            self.enorm = None

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * \
            W, f"input feature has wrong size, with L = {L}, H = {H}, W = {W}"

        shortcut = x

        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x,
                                   shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        orig_type = x.dtype  # attn may force to fp32
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp,
                                   Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        if self.mlpfp32:
            x = self.norm1.float()(x)
            x = x.type(orig_type)
        else:
            x = self.norm1(x)
        x = shortcut + self.drop_path(x)
        shortcut = x

        orig_type = x.dtype
        x = self.mlp(x, H, W)
        if self.mlpfp32:
            x = self.norm2.float()(x)
            x = x.type(orig_type)
        else:
            x = self.norm2(x)
        x = shortcut + self.drop_path(x)

        if self.endnorm:
            x = self.enorm(x)

        return x


class SwinTransformerBlockPre(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 use_mlp_norm=False,
                 endnorm=False,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init_values=None,
                 relative_coords_table_type='norm8_log',
                 rpe_hidden_dim=512,
                 rpe_output_type='normal',
                 attn_type='normal',
                 mlp_type='normal',
                 mlpfp32=False,
                 pretrain_window_size=-1,
                 head_chunk_size=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_mlp_norm = use_mlp_norm
        self.endnorm = endnorm
        self.mlpfp32 = mlpfp32
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            relative_coords_table_type=relative_coords_table_type,
            rpe_output_type=rpe_output_type,
            rpe_hidden_dim=rpe_hidden_dim,
            attn_type=attn_type,
            mlpfp32=mlpfp32,
            pretrain_window_size=pretrain_window_size,
            head_chunk_size=head_chunk_size)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if mlp_type == 'normal':
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                norm_layer=norm_layer if self.use_mlp_norm else None,
                mlpfp32=mlpfp32)
        elif mlp_type == 'conv':
            self.mlp = ConvMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                norm_layer=norm_layer if self.use_mlp_norm else None,
                mlpfp32=mlpfp32)
        elif mlp_type == 'conv_ln':
            self.mlp = ConvMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                norm_layer=norm_layer if self.use_mlp_norm else None,
                mlpfp32=mlpfp32,
                proj_ln=True)

        if init_values is not None and init_values >= 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),
                                        requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),
                                        requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = 1.0, 1.0

        if self.endnorm:
            self.enorm = norm_layer(dim)
        else:
            self.enorm = None

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * \
            W, f"input feature has wrong size, with L = {L}, H = {H}, W = {W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x,
                                   shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        orig_type = x.dtype  # attn may force to fp32
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp,
                                   Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        if self.mlpfp32:
            x = self.gamma_1 * x
            x = x.type(orig_type)
        else:
            x = self.gamma_1 * x
        x = shortcut + self.drop_path(x)
        shortcut = x

        orig_type = x.dtype
        x = self.norm2(x)
        if self.mlpfp32:
            x = self.gamma_2 * self.mlp(x, H, W)
            x = x.type(orig_type)
        else:
            x = self.gamma_2 * self.mlp(x, H, W)
        x = shortcut + self.drop_path(x)

        if self.endnorm:
            x = self.enorm(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, postnorm=True):
        super().__init__()
        self.dim = dim
        self.postnorm = postnorm

        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim) if postnorm else norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        if self.postnorm:
            x = self.reduction(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.reduction(x)

        return x


class PatchReduction1C(nn.Module):
    r""" Patch Reduction Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, postnorm=True):
        super().__init__()
        self.dim = dim
        self.postnorm = postnorm

        self.reduction = nn.Linear(dim, dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        if self.postnorm:
            x = self.reduction(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.reduction(x)

        return x


class ConvPatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, postnorm=True):
        super().__init__()
        self.dim = dim
        self.postnorm = postnorm

        self.reduction = nn.Conv2d(dim,
                                   2 * dim,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1)
        self.norm = norm_layer(2 * dim) if postnorm else norm_layer(dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        if self.postnorm:
            x = x.permute(0, 3, 1, 2)  # B C H W
            x = self.reduction(x).flatten(2).transpose(1, 2)  # B H//2*W//2 2*C
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)  # B C H W
            x = self.reduction(x).flatten(2).transpose(1, 2)  # B H//2*W//2 2*C

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        use_shift (bool): Whether to use shifted window. Default: True.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 checkpoint_blocks=255,
                 init_values=None,
                 endnorm_interval=-1,
                 use_mlp_norm=False,
                 use_shift=True,
                 relative_coords_table_type='norm8_log',
                 rpe_hidden_dim=512,
                 rpe_output_type='normal',
                 attn_type='normal',
                 mlp_type='normal',
                 mlpfp32_blocks=[-1],
                 postnorm=True,
                 pretrain_window_size=-1,
                 head_chunk_size=1):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.checkpoint_blocks = checkpoint_blocks
        self.init_values = init_values if init_values is not None else 0.0
        self.endnorm_interval = endnorm_interval
        self.mlpfp32_blocks = mlpfp32_blocks
        self.postnorm = postnorm

        # build blocks
        if self.postnorm:
            self.blocks = nn.ModuleList([
                SwinTransformerBlockPost(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if
                    (i % 2 == 0) or (not use_shift) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_mlp_norm=use_mlp_norm,
                    endnorm=True if ((i + 1) % endnorm_interval == 0) and
                    (endnorm_interval > 0) else False,
                    relative_coords_table_type=relative_coords_table_type,
                    rpe_hidden_dim=rpe_hidden_dim,
                    rpe_output_type=rpe_output_type,
                    attn_type=attn_type,
                    mlp_type=mlp_type,
                    mlpfp32=True if i in mlpfp32_blocks else False,
                    pretrain_window_size=pretrain_window_size,
                    head_chunk_size=head_chunk_size) for i in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                SwinTransformerBlockPre(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if
                    (i % 2 == 0) or (not use_shift) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    init_values=init_values,
                    use_mlp_norm=use_mlp_norm,
                    endnorm=True if ((i + 1) % endnorm_interval == 0) and
                    (endnorm_interval > 0) else False,
                    relative_coords_table_type=relative_coords_table_type,
                    rpe_hidden_dim=rpe_hidden_dim,
                    rpe_output_type=rpe_output_type,
                    attn_type=attn_type,
                    mlp_type=mlp_type,
                    mlpfp32=True if i in mlpfp32_blocks else False,
                    pretrain_window_size=pretrain_window_size,
                    head_chunk_size=head_chunk_size) for i in range(depth)
            ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim,
                                         norm_layer=norm_layer,
                                         postnorm=postnorm)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size,
                          -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size,
                          -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # nW, window_size, window_size, 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                          float(-100.0)).masked_fill(
                                              attn_mask == 0, float(0.0))
        for idx, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            if isinstance(self.downsample, PatchReduction1C):
                return x, H, W, x_down, H, W
            else:
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
                return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W

    def _init_block_norm_weights(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, self.init_values)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, self.init_values)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x,
                      (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class ResNetDLNPatchEmbed(nn.Module):

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(4)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, stride=2, padding=1, bias=False),
            LayerNorm2D(64, norm_layer), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            LayerNorm2D(64, norm_layer), nn.GELU(),
            nn.Conv2d(64, embed_dim, 3, stride=1, padding=1, bias=False))
        self.norm = LayerNorm2D(embed_dim, norm_layer if norm_layer is not None
                                else LayerNormFP32)  # use ln always
        self.act = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x,
                      (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.maxpool(x)
        # x = x.flatten(2).transpose(1, 2)
        return x


@MODELS.register_module()
class SwinV2TransformerRPE2FC(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        use_shift (bool): Whether to use shifted window. Default: True.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=partial(LayerNormFP32, eps=1e-6),
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 init_values=1e-5,
                 endnorm_interval=-1,
                 use_mlp_norm_layers=[],
                 relative_coords_table_type='norm8_log_bylayer',
                 rpe_hidden_dim=512,
                 attn_type='cosine_mh',
                 rpe_output_type='sigmoid',
                 rpe_wd=False,
                 postnorm=True,
                 mlp_type='normal',
                 patch_embed_type='normal',
                 patch_merge_type='normal',
                 strid16=False,
                 checkpoint_blocks=[255, 255, 255, 255],
                 mlpfp32_layer_blocks=[[-1], [-1], [-1], [-1]],
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_shift=True,
                 rpe_interpolation='geo',
                 pretrain_window_size=[-1, -1, -1, -1],
                 init_cfg=None,
                 head_chunk_size=1):

        super().__init__()

        self.init_cfg = init_cfg
        self.pretrain_img_size = pretrain_img_size
        self.depths = depths
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.rpe_interpolation = rpe_interpolation
        self.mlp_ratio = mlp_ratio
        self.endnorm_interval = endnorm_interval
        self.use_mlp_norm_layers = use_mlp_norm_layers
        self.relative_coords_table_type = relative_coords_table_type
        self.rpe_hidden_dim = rpe_hidden_dim
        self.rpe_output_type = rpe_output_type
        self.rpe_wd = rpe_wd
        self.attn_type = attn_type
        self.postnorm = postnorm
        self.mlp_type = mlp_type
        self.strid16 = strid16

        if isinstance(window_size, list):
            pass
        elif isinstance(window_size, int):
            window_size = [window_size] * self.num_layers
        else:
            raise TypeError("We only support list or int for window size")

        if isinstance(use_shift, list):
            pass
        elif isinstance(use_shift, bool):
            use_shift = [use_shift] * self.num_layers
        else:
            raise TypeError("We only support list or bool for use_shift")

        if isinstance(use_checkpoint, list):
            pass
        elif isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint] * self.num_layers
        else:
            raise TypeError("We only support list or bool for use_checkpoint")

        # split image into non-overlapping patches
        if patch_embed_type == 'normal':
            self.patch_embed = PatchEmbed(
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
        elif patch_embed_type == 'resnetdln':
            assert patch_size == 4, "check"
            self.patch_embed = ResNetDLNPatchEmbed(in_chans=in_chans,
                                                   embed_dim=embed_dim,
                                                   norm_layer=norm_layer)
        elif patch_embed_type == 'resnetdnf':
            assert patch_size == 4, "check"
            self.patch_embed = ResNetDLNPatchEmbed(in_chans=in_chans,
                                                   embed_dim=embed_dim,
                                                   norm_layer=None)
        else:
            raise NotImplementedError()
        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1]
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0],
                            patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        if patch_merge_type == 'normal':
            downsample_layer = PatchMerging
        elif patch_merge_type == 'conv':
            downsample_layer = ConvPatchMerging
        else:
            raise NotImplementedError()
        # build layers
        self.layers = nn.ModuleList()
        num_features = []
        for i_layer in range(self.num_layers):
            cur_dim = int(embed_dim * 2 ** (i_layer - 1)) \
                if (i_layer == self.num_layers - 1 and strid16) else \
                int(embed_dim * 2 ** i_layer)
            num_features.append(cur_dim)
            if i_layer < self.num_layers - 2:
                cur_downsample_layer = downsample_layer
            elif i_layer == self.num_layers - 2:
                if strid16:
                    cur_downsample_layer = PatchReduction1C
                else:
                    cur_downsample_layer = downsample_layer
            else:
                cur_downsample_layer = None
            layer = BasicLayer(
                dim=cur_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=cur_downsample_layer,
                use_checkpoint=use_checkpoint[i_layer],
                checkpoint_blocks=checkpoint_blocks[i_layer],
                init_values=init_values,
                endnorm_interval=endnorm_interval,
                use_mlp_norm=True if i_layer in use_mlp_norm_layers else False,
                use_shift=use_shift[i_layer],
                relative_coords_table_type=self.relative_coords_table_type,
                rpe_hidden_dim=self.rpe_hidden_dim,
                rpe_output_type=self.rpe_output_type,
                attn_type=self.attn_type,
                mlp_type=self.mlp_type,
                mlpfp32_blocks=mlpfp32_layer_blocks[i_layer],
                postnorm=self.postnorm,
                pretrain_window_size=pretrain_window_size[i_layer],
                head_chunk_size=head_chunk_size)
            self.layers.append(layer)

        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(self.init_cfg.checkpoint,
                                    logger=logger,
                                    map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                if k.startswith('encoder.'):
                    state_dict[k[8:]] = v
            if len(state_dict) == 0:
                state_dict = OrderedDict(_state_dict)
            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            state_dict = {
                k: v
                for k, v in state_dict.items()
                if ('relative_coords_table' not in k) and (
                    'relative_position_index' not in k)
            }
            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed,
                                               size=(Wh, Ww),
                                               mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1,
                                                              2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer.float()(x_out.float())

                out = x_out.view(-1, H, W,
                                 self.num_features[i]).permute(0, 3, 1,
                                                               2).contiguous()
                outs.append(out)

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinV2TransformerRPE2FC, self).train(mode)
        self._freeze_stages()


#=============================================================================#
#                        ait/code/model/transformer.py                        #
#=============================================================================#


import torch
import torch.nn.functional as F
from torch import nn
from .utils import MODELS

from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_
import torch.utils.checkpoint as checkpoint
import math
from mmcv.runner import force_fp32, BaseModule, auto_fp16


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def with_pos(self, x, pos):
        return x if pos is None else x + pos

    def forward(self, q, k, v, mask=None, pre_kv=None, qpos=None, kpos=None):
        B, N, C = q.shape
        q = self.wq(self.with_pos(q, qpos)).reshape(
            B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.wk(self.with_pos(k, kpos)).reshape(
            B, k.shape[1], self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.wv(v).reshape(
            B, v.shape[1], self.num_heads, C // self.num_heads).transpose(1, 2)

        if pre_kv is not None:
            k = torch.cat([pre_kv[0], k], dim=2)
            v = torch.cat([pre_kv[1], v], dim=2)
            pre_kv = torch.stack([k, v], dim=0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn.masked_fill_(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x if pre_kv is None else (x, pre_kv)


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, drop_path=0.1, drop_out=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=(drop_out, 0.))

    def forward(self, x, mask=None, pos=None):
        norm_x = self.norm1(x)
        x = x + self.drop_path(self.attn(norm_x, norm_x,
                               norm_x, mask, qpos=pos, kpos=pos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, drop_path=0.1, drop_out=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        self.crossattn = Attention(dim, num_heads=num_heads)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.dropout = nn.Dropout(drop_out)
        self.norm3 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=(drop_out, 0.))

    def forward(self, x, z, mask1=None, mask2=None, pos=None, zpos=None, pre_kv=None):
        norm_x = self.norm1(x)
        if pre_kv is None:
            y = self.attn(norm_x, norm_x, norm_x, mask1, qpos=pos, kpos=pos)
        else:
            pos = None if pos is None else pos[:, -1:, :]
            y, pre_kv = self.attn(norm_x, norm_x, norm_x,
                                  mask1, pre_kv=pre_kv, qpos=pos, kpos=pos)
        x = x + self.drop_path(y)
        norm_x = self.norm2(x)
        norm_z = self.norm2(z)
        x = x + self.drop_path(self.crossattn(norm_x,
                               norm_z, norm_z, mask2, qpos=pos, kpos=zpos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x if pre_kv is None else (x, pre_kv)


class Sequential(nn.Module):
    def __init__(self, *blocks, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.use_checkpoint = use_checkpoint

    def __len__(self):
        return len(self.blocks)

    def forward(self, x, pre_kv_list=None, *args, **kwargs):
        use_checkpoint = self.use_checkpoint and self.training
        if pre_kv_list is None:
            for blk in self.blocks:
                if use_checkpoint:
                    assert len(kwargs) == 0
                    x = checkpoint.checkpoint(blk, x, *args, **kwargs)
                else:
                    x = blk(x, *args, **kwargs)
            return x
        else:
            cur_kv_list = []  # only use in eval
            for blk, pre_kv in zip(self.blocks, pre_kv_list):
                x, cur_kv = blk(x, *args, pre_kv=pre_kv, **kwargs)
                cur_kv_list.append(cur_kv)
            return x, cur_kv_list


@MODELS.register_module()
class ARTransformer(nn.Module):
    def __init__(self, in_chans, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, drop_path=0.1, drop_out=0.1, drop_path_linear=False,
                 num_vocal=2094, dec_length=2100, pred_eos=False,
                 num_bins=2001, num_classes=80, num_embeddings=128, num_embeddings_depth=128, checkpoint_encoder=False, checkpoint_decoder=False,
                 top_p=0.4, delay_eos=0, qk_pos=False, enc_mask=False, dec_mask=False,
                 pos_enc='sine', n_rows=20, n_cols=20, mask_before_label=False, ntasks=4, with_dec_embed=True, soft_vae=False, soft_transformer=False,
                 parallel=False, head_args={}
                 ):
        super().__init__()
        self.fp16_enabled = False
        self.head_args = head_args
        self.parallel = parallel
        self.soft_vae = soft_vae
        self.soft_transformer = soft_transformer
        self.num_vocal = num_vocal
        self.pred_eos = pred_eos
        self.top_p = top_p
        self.delay_eos = delay_eos
        self.qk_pos = qk_pos
        self.enc_mask = enc_mask
        self.dec_mask = dec_mask
        self.num_classes = num_classes
        self.mask_before_label = mask_before_label

        self.num_bins = num_bins
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.num_embeddings_depth = num_embeddings_depth

        self.class_offset = num_bins + 1
        self.special_offset = self.class_offset + num_classes
        self.noise_label = self.special_offset + 1
        self.mask_offset = self.special_offset + 2
        self.depth_offset = self.mask_offset + num_embeddings
        self.enddepth_offset = self.depth_offset + num_embeddings_depth

        self.nhead = nhead
        self.d_model = d_model

        self.n_rows = n_rows
        self.n_cols = n_cols

        self.add_enc_embed(pos_enc, n_rows, n_cols, d_model)

        if self.parallel:
            self.mask_embed = nn.Parameter(torch.empty(1, 1, d_model))
        self.det_embed = nn.Parameter(torch.empty(ntasks, 1, d_model))
        self.voc_embed = nn.Parameter(torch.empty(
            self.num_vocal - 2, d_model)) if pred_eos else nn.Parameter(torch.empty(self.num_vocal, d_model))
        self.with_dec_embed = with_dec_embed
        if with_dec_embed:
            self.dec_embed = nn.Parameter(
                torch.empty(ntasks, dec_length, d_model))

        self.input_proj = nn.Linear(in_chans, d_model)
        dpr = iter(torch.linspace(0, drop_path, num_encoder_layers).tolist(
        )) if drop_path_linear else iter([drop_path]*num_encoder_layers)
        self.encoder = Sequential(*[
            EncoderBlock(d_model, nhead, dim_feedforward,
                         drop_path=next(dpr), drop_out=drop_out)
            for _ in range(num_encoder_layers)
        ], use_checkpoint=checkpoint_encoder)
        dpr = iter(torch.linspace(0, drop_path, num_decoder_layers).tolist(
        )) if drop_path_linear else iter([drop_path]*num_decoder_layers)
        self.decoder = Sequential(*[
            DecoderBlock(d_model, nhead, dim_feedforward,
                         drop_path=next(dpr), drop_out=drop_out)
            for i in range(num_decoder_layers)
        ], use_checkpoint=checkpoint_decoder)

        self.norm = nn.LayerNorm(d_model)

        # self.vocal_classifier = nn.Linear(d_model, num_vocal)
        self.outp_bias = nn.Parameter(torch.empty(num_vocal))

        self.dropout = nn.Dropout(drop_out)
        self.stem_ln = nn.LayerNorm(d_model)
        self.encout_ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_ln = nn.LayerNorm(d_model)
        self.proj_mlp = Mlp(
            in_features=d_model, hidden_features=dim_feedforward, drop=(drop_out, 0.))
        self.proj_mlp_ln = nn.LayerNorm(d_model)
        self.proj_mlp_droppath = DropPath(drop_path)

        self.init_weights()

    def vocal_classifier(self, x):
        return x @ self.voc_embed.transpose(0, 1) + self.outp_bias

    def init_weights(self):
        trunc_normal_(self.det_embed, std=0.02)
        trunc_normal_(self.voc_embed, std=0.02)
        if hasattr(self, 'mask_embed'):
            trunc_normal_(self.mask_embed, std=0.02)
        if self.with_dec_embed:
            trunc_normal_(self.dec_embed, std=0.02)
        trunc_normal_(self.outp_bias, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def add_enc_embed(self, pos_enc, n_rows, n_cols, d_model):
        if pos_enc == 'sine':
            d_model /= 2
            y_embed = torch.arange(n_rows, dtype=torch.float32)
            x_embed = torch.arange(n_cols, dtype=torch.float32)
            dim_t = torch.arange(
                d_model, dtype=torch.float32)
            dim_t = 10000. ** (2 * (dim_t // 2) / d_model)
            pos_x = x_embed[:, None] / dim_t
            pos_y = y_embed[:, None] / dim_t
            pos_x = torch.stack(
                (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()),
                dim=-1).view(1, n_cols, -1).expand(n_rows, n_cols, -1)
            pos_y = torch.stack(
                (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()),
                dim=-1).view(n_rows, 1, -1).expand(n_rows, n_cols, -1)
            pos = torch.cat((pos_y, pos_x), dim=-1).view(1, n_rows, n_cols, -1)
            self.register_buffer("enc_embed", pos)
        elif pos_enc == 'sine_norm':
            d_model /= 2
            y_embed = torch.arange(n_rows, dtype=torch.float32)
            x_embed = torch.arange(n_cols, dtype=torch.float32)
            y_embed = y_embed / (n_rows-1) * 2 * math.pi
            x_embed = x_embed / (n_cols-1) * 2 * math.pi
            dim_t = torch.arange(
                d_model, dtype=torch.float32)
            dim_t = 10000. ** (2 * (dim_t // 2) / d_model)
            pos_x = x_embed[:, None] / dim_t
            pos_y = y_embed[:, None] / dim_t
            pos_x = torch.stack(
                (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()),
                dim=-1).view(1, n_cols, -1).expand(n_rows, n_cols, -1)
            pos_y = torch.stack(
                (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()),
                dim=-1).view(n_rows, 1, -1).expand(n_rows, n_cols, -1)
            pos = torch.cat((pos_y, pos_x), dim=-1).view(1, n_rows, n_cols, -1)
            self.register_buffer("enc_embed", pos)
        elif pos_enc == 'learned':
            self.enc_embed = nn.Parameter(
                torch.empty(1, n_rows, n_cols, d_model))
            trunc_normal_(self.enc_embed, std=0.02)
        else:
            raise ValueError('Unknown pos encoding %s' % pos_enc)

    @force_fp32()
    def forward(self, src, input_seq, mask, task_id, pred_len=0):
        """
        Args:
            src: shape[B, C, H, W]
            input_seq: shape[B, 501, C] for training and shape[B, 1, C] for inference
            mask: shape[B, H, W]
            pred_len is used for test only
        """
        H, W = src.shape[-2:]
        src = self.input_proj(self.dropout(src.flatten(2).transpose(1, 2)))
        src = self.stem_ln(src)
        mask = mask.flatten(1)[:, None, None, :] if (
            self.enc_mask or self.dec_mask) else None
        enc_mask = mask if self.enc_mask else None

        B, N, C = src.shape
        enc_embed = self.enc_embed[:, :H, :W, :].flatten(1, 2)
        z = self.encoder(src, None, enc_mask, enc_embed) \
            if self.qk_pos else self.encoder(src + enc_embed, None, enc_mask)
        z = self.encout_ln(z)
        # Add (optional) positional embedding to encoded visual units.
        z = self.proj_ln(self.proj(z)) + enc_embed
        z = z + self.proj_mlp_droppath(self.proj_mlp(self.proj_mlp_ln(z)))

        dec_mask = mask if self.dec_mask else None
        if self.training:
            # if task_id != 3:
            B = input_seq.shape[0]
            if z.shape[0] != B:
                z = z.repeat_interleave(B // z.shape[0], dim=0)
            M = input_seq.shape[1] + 1
            if self.with_dec_embed:
                pos = self.dec_embed[task_id, :M, :]
            input_embed = self.voc_embed[input_seq]
            if self.parallel:
                assert task_id[0] == 2
                input_embed = self.mask_embed.expand_as(input_embed)

            input_embed = torch.cat([
                self.det_embed[task_id, :, :],
                input_embed,
            ], dim=1)
            self_attn_mask = torch.triu(torch.ones(
                M, M, device=z.device), diagonal=1).bool()

            x = self.decoder(
                input_embed, None, z, self_attn_mask, dec_mask
            ) if not self.with_dec_embed else self.decoder(
                input_embed, None, z, self_attn_mask, dec_mask, pos, enc_embed
            ) if self.qk_pos else self.decoder(
                input_embed + pos, None, z, self_attn_mask, dec_mask
            )

            x = self.norm(x)
            pred_seq_logits = self.vocal_classifier(x)
            return pred_seq_logits
        else:
            if task_id == 0 or task_id == 1:  # det or insseg
                mask_token_length = 0 if task_id == 0 else self.mask_token_length  # got by insseghead
                end = torch.zeros(B, device=z.device).bool()
                end_lens = torch.zeros(B, device=z.device).long()
                obj_len = 5 + mask_token_length

                input_embed = self.det_embed[[task_id]].expand(B, -1, -1)
                pre_kv_lsit = [
                    torch.empty(
                        (2, B, self.nhead, 0, self.d_model // self.nhead),
                        device=z.device, dtype=torch.float32
                    )
                    for _ in range(len(self.decoder))
                ]
                pred_tokens = []
                pred_scores = []
                pred_mask_logits = []
                pred_box_logits = []

                def is_label(i):
                    if self.mask_before_label:
                        return (i-1) % obj_len == obj_len-1
                    else:
                        return (i-1) % obj_len == 4
                for i in range(1, pred_len + 1):
                    if self.with_dec_embed:
                        pos = self.dec_embed[[task_id], :i, :]
                    x, pre_kv_lsit = self.decoder(
                        input_embed, pre_kv_lsit, z, None, dec_mask
                    ) if not self.with_dec_embed else self.decoder(
                        input_embed, pre_kv_lsit, z, None, dec_mask, pos, enc_embed
                    ) if self.qk_pos else self.decoder(
                        input_embed + pos[:, -1:,
                                          :], pre_kv_lsit, z, None, dec_mask
                    )
                    x = self.norm(x)
                    logits = self.vocal_classifier(x)[:, -1, :]

                    is_mask_flg = False
                    is_label_flg = False
                    is_box_flg = False
                    if is_label(i):  # label
                        is_label_flg = True
                        offset = self.class_offset
                        offset_end = self.class_offset + self.num_classes
                        current_logits = logits[:,
                                                self.class_offset: self.class_offset + self.num_classes]
                        if self.pred_eos:
                            current_scores = current_logits.softmax(dim=-1)
                        else:
                            current_scores = torch.cat([current_logits, logits[:, [self.noise_label]]], dim=1).softmax(
                                dim=-1)[:, :-1]  # add noise label
                    elif (i-1) % obj_len < 4:  # box
                        is_box_flg = True
                        offset = 0
                        offset_end = self.num_bins+1
                        current_logits = logits[:, :self.num_bins+1]
                    else:  # mask
                        is_mask_flg = True
                        offset = self.mask_offset
                        offset_end = self.mask_offset+self.num_embeddings
                        current_logits = logits[:,
                                                self.mask_offset: self.mask_offset+self.num_embeddings]
                        if self.soft_vae or self.soft_transformer:
                            tmp_current_logits = current_logits.clone()
                            pred_mask_logits.append(tmp_current_logits)

                    top_p = self.top_p

                    if self.pred_eos and (i - 1) % obj_len == 0:
                        current_logits = torch.cat(
                            [current_logits, logits[:, [self.special_offset+1]] - self.delay_eos], dim=1)

                    # Sort logits in descending order to determine the nucleus.
                    sorted_logits, sorted_idx = torch.sort(
                        current_logits, descending=True)

                    # Get cumulative softmax probabilites. For every instance in batch, a
                    #  variable amount of tokens (N) will consitute the nucleus.
                    # shape: (batch_size, num_classes)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Determine indices of tokens at the tail of distribution. These will be
                    # removed from the nucleus.
                    sorted_idx_to_remove = cumulative_probs > top_p

                    # Shift the indices to the right to keep the first token outside nucleus.
                    sorted_idx_to_remove[...,
                                         1:] = sorted_idx_to_remove[..., :-1].clone()
                    sorted_idx_to_remove[..., 0] = 0

                    # Set logits to large negative value to avoid sampling them. Iterate over
                    # the batch of examples.
                    for t in range(current_logits.size()[0]):
                        idx_to_remove = sorted_idx[t][sorted_idx_to_remove[t]]
                        current_logits[t][idx_to_remove] = -float('Inf')

                    # Sample from the filtered distribution.
                    # shape: (batch_size, num_classes)
                    current_probs = F.softmax(current_logits, dim=-1)

                    # shape: (batch_size, )
                    current_predictions = torch.multinomial(current_probs, 1)
                    current_predictions = current_predictions.view(B)

                    if self.pred_eos and (i - 1) % obj_len == 0:
                        stop_state = current_predictions.eq(
                            current_logits.shape[-1] - 1)
                        end_lens += i * (~end * stop_state)
                        end = (stop_state + end).bool()
                        if end.all():
                            break
                    pred_token = current_predictions[:, None]
                    if is_label(i):  # label
                        pred_scores.append(torch.gather(
                            current_scores, 1, pred_token))

                    if self.soft_transformer and is_mask_flg:
                        input_embed = tmp_current_logits.softmax(
                            dim=1) @ self.voc_embed[offset: offset_end]
                        input_embed = input_embed.unsqueeze(1)
                    else:
                        input_embed = self.voc_embed[(pred_token + offset)]
                    pred_tokens.append(pred_token)

                if not self.pred_eos:
                    end_lens.fill_(pred_len)
                else:
                    end_lens[end_lens == 0] = self.dec_embed.size(1) - 1
                pred_tokens = torch.cat(pred_tokens, dim=1) \
                    if len(pred_tokens) > 0 else torch.tensor([], device=src.device).view(B, 0)
                pred_scores = torch.cat(pred_scores, dim=1) \
                    if len(pred_scores) > 0 else torch.tensor([], device=src.device).view(B, 0)

                if self.soft_vae:
                    pred_mask_logits = torch.stack(pred_mask_logits, dim=1)
                    Bs, L, C = pred_mask_logits.shape
                    pred_mask_logits = pred_mask_logits.view(
                        Bs, -1, mask_token_length, C)
                    pred_tokens = [(psl[:end_idx], score, mask_logit) for end_idx, psl, score, mask_logit in zip(
                        end_lens, pred_tokens, pred_scores, pred_mask_logits)]
                else:
                    pred_tokens = [(psl[:end_idx], score) for end_idx, psl, score in zip(
                        end_lens, pred_tokens, pred_scores)]
                return pred_tokens

            elif task_id == 2:  # depth
                input_embed = self.det_embed[[task_id]].expand(B, -1, -1)
                pre_kv_lsit = [
                    torch.empty(
                        (2, B, self.nhead, 0, self.d_model // self.nhead),
                        device=z.device, dtype=torch.float32
                    )
                    for _ in range(len(self.decoder))
                ]
                pred_tokens = []
                pred_logits = []
                for i in range(1, pred_len + 1):
                    if self.with_dec_embed:
                        pos = self.dec_embed[[task_id], :i, :]
                    x, pre_kv_lsit = self.decoder(
                        input_embed, pre_kv_lsit, z, None, dec_mask
                    ) if not self.with_dec_embed else self.decoder(
                        input_embed, pre_kv_lsit, z, None, dec_mask, pos, self.enc_embed
                    ) if self.qk_pos else self.decoder(
                        input_embed + pos[:, -1:,
                                          :], pre_kv_lsit, z, None, dec_mask
                    )
                    x = self.norm(x)
                    logits = self.vocal_classifier(x)[:, -1, :]

                    # varify prob
                    # varlogits, varidx = torch.sort(logits.softmax(dim=1), dim=1, descending=True)
                    # varifycumsum(varlogits, varidx, self.depth_offset-2, self.num_vocal-2)

                    current_logits = logits[:,
                                            self.depth_offset: self.enddepth_offset]
                    tmp_logits = current_logits.clone()

                    # Sort logits in descending order to determine the nucleus.
                    sorted_logits, sorted_idx = torch.sort(
                        current_logits, descending=True)

                    # Get cumulative softmax probabilites. For every instance in batch, a
                    #  variable amount of tokens (N) will consitute the nucleus.
                    # shape: (batch_size, num_classes)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Determine indices of tokens at the tail of distribution. These will be
                    # removed from the nucleus.
                    sorted_idx_to_remove = cumulative_probs > self.top_p

                    # Shift the indices to the right to keep the first token outside nucleus.
                    sorted_idx_to_remove[...,
                                         1:] = sorted_idx_to_remove[..., :-1].clone()
                    sorted_idx_to_remove[..., 0] = 0

                    # Set logits to large negative value to avoid sampling them. Iterate over
                    # the batch of examples.
                    for t in range(current_logits.size()[0]):
                        idx_to_remove = sorted_idx[t][sorted_idx_to_remove[t]]
                        current_logits[t][idx_to_remove] = -float('Inf')

                    # Sample from the filtered distribution.
                    # shape: (batch_size, num_classes)
                    current_probs = F.softmax(current_logits, dim=-1)

                    # shape: (batch_size, )
                    current_predictions = torch.multinomial(current_probs, 1)
                    current_predictions = current_predictions.view(B)

                    pred_token = current_predictions[:, None]
                    pred_tokens.append(current_predictions)
                    pred_logits.append(tmp_logits)

                    if self.parallel:
                        # TODO: rewrite to actually parallel to accelerate code
                        input_embed = self.mask_embed.expand(B, -1, -1)
                    elif self.soft_transformer:
                        input_embed = tmp_logits.softmax(
                            dim=1) @ self.voc_embed[self.depth_offset: self.enddepth_offset]
                        input_embed = input_embed.unsqueeze(1)
                    else:
                        input_embed = self.voc_embed[pred_token +
                                                     self.depth_offset]

                pred_logits = torch.stack(pred_logits, dim=1)
                pred_tokens = torch.stack(pred_tokens, dim=1)
                return pred_logits, pred_tokens

            else:
                raise NotImplementedError


#=============================================================================#
#                           ait/code/model/vqvae.py                           #
#=============================================================================#


import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from math import log2, sqrt
from .utils import MODELS


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 /
                                             self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs, return_indices=False):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        if return_indices:
            return encoding_indices.squeeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                               torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.register_buffer('_embedding', torch.empty(
            self._num_embeddings, self._embedding_dim))
        self._embedding.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.empty(
            num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs, return_indices=False):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        if return_indices:
            return encoding_indices.squeeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input.detach())
            self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw

            self._embedding = self._ema_w / self._ema_cluster_size.unsqueeze(1)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                               torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, encoding_indices.view(input_shape[0:3])


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class ResBlock_v1(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class ResBlock_v2(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1, bias=False),
            nn.BatchNorm2d(chan),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1, bias=False),
            nn.BatchNorm2d(chan),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)


@MODELS.register_module()
class VQVAE(nn.Module):
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if 'codebook.weight' in state_dict:
            del state_dict['codebook.weight']

    def __init__(self, token_length=16, mask_size=64, embedding_dim=512, num_embeddings=1024, pretrained='vae.pt', freeze=True, num_resnet_blocks=2, hidden_dim=256, use_norm=True, use_sigmoid=False, tau=1.0):
        # adapt param
        self.tau = tau
        self.token_length = token_length
        image_size = mask_size
        num_tokens = num_embeddings
        codebook_dim = embedding_dim
        channels = 1
        loss_type = 'mse'
        temperature = 0.9
        straight_through = False
        kl_div_loss_weight = 0.
        simplify = False
        max_value = 1.0
        use_softmax = False
        determistic = False
        downsample_ratio = mask_size // int(sqrt(token_length))
        train_objective = "regression"
        residul_type = 'v1'
        ema_decay = 0.99
        commitment_cost = 0.25
        super().__init__()
        # assert log2(image_size).is_integer(), 'image size must be a power of 2'
        hidden_dim = hidden_dim // (downsample_ratio // 2)
        self.num_resnet_blocks = num_resnet_blocks
        self.simplify = simplify
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.temperature = temperature
        self.straight_through = straight_through
        self.use_softmax = use_softmax
        self.determistic = determistic
        self.downsample_ratio = downsample_ratio
        self.layer_num = int(log2(downsample_ratio) - 1)
        self.use_norm = use_norm
        self.use_sigmoid = use_sigmoid
        self.train_objective = train_objective
        self.max_value = max_value
        self.residul_type = residul_type
        if self.residul_type == 'v1':
            ResBlock = ResBlock_v1
        elif self.residul_type == 'v2':
            ResBlock = ResBlock_v2

        dim = hidden_dim
        enc_layers = [
            nn.Conv2d(channels, dim, 4, stride=2, padding=1), nn.ReLU()]

        for i in range(self.layer_num):
            enc_layers.append(nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1))
            enc_layers.append(nn.ReLU())
            dim = dim * 2

        for i in range(num_resnet_blocks):
            enc_layers.append(ResBlock(dim))
        enc_layers.append(nn.Conv2d(dim, codebook_dim, 1))

        dim = hidden_dim * self.downsample_ratio // 2

        dec_layers = [nn.Conv2d(codebook_dim, dim, 1), nn.ReLU()]

        for i in range(num_resnet_blocks):
            dec_layers.append(ResBlock(dim))

        dec_layers.append(nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1))
        dec_layers.append(nn.ReLU())
        for i in range(self.layer_num):
            dec_layers.append(nn.ConvTranspose2d(
                dim, dim // 2, 4, stride=2, padding=1))
            dec_layers.append(nn.ReLU())
            dim = dim // 2
        dec_layers.append(nn.Conv2d(dim, channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        if loss_type == 'smooth_l1':
            self.loss_fn = F.smooth_l1_loss
        elif loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'mse':
            self.loss_fn = F.mse_loss
        elif loss_type == 'cross_entropy':
            assert self.train_objective == 'classification'
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            assert "loss_type {0} is not implemented".format(loss_type)

        self.kl_div_loss_weight = kl_div_loss_weight

        if ema_decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_tokens, codebook_dim,
                                              commitment_cost, ema_decay)
        else:
            self._vq_vae = VectorQuantizer(num_tokens, codebook_dim,
                                           commitment_cost)

        assert pretrained is not None
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'weights' in state_dict:
            state_dict = state_dict['weights']
        if 'module' in state_dict:
            state_dict = state_dict['module']
        self.load_state_dict(state_dict, strict=True)
        if freeze:
            self.freeze_layer()

    def freeze_layer(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def init_weights(self, pretrained=None):
        # assert pretrained == Nones
        if pretrained is not None:
            pretrained_model = torch.load(pretrained)
            self.load_state_dict(pretrained_model['weights'])

    def norm(self, images):
        images = 2. * (images / self.max_value - 0.5)
        return images

    def denorm(self, images):
        images = images * 0.5 + 0.5
        return images * self.max_value

    @torch.no_grad()
    @eval_decorator
    def encode(
        self,
        img,
        use_norm=None
    ):
        B = img.shape[0]
        image_size = self.image_size
        assert img.shape[-1] == image_size and img.shape[
            -2] == image_size, f'input must have the correct image size {image_size}'
        img = img.float()

        use_norm = self.use_norm if use_norm is None else use_norm
        if use_norm:
            img = self.norm(img)

        logits = self.encoder(img.unsqueeze(1))
        return self._vq_vae(logits, return_indices=True).view(B, self.token_length)

    def decode(
        self,
        img_seq
    ):
        image_embeds = self._vq_vae._embedding[img_seq]
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))
        image_embeds = image_embeds.view(b, h, w, d)

        image_embeds = rearrange(image_embeds, 'b h w d -> b d h w', h=h, w=w)
        images = self.decoder(image_embeds)

        if self.train_objective == 'classification':
            images = images.argmax(dim=1, keepdim=True)
        if self.use_sigmoid:
            return images.sigmoid().squeeze(1)
        elif self.use_norm:
            return self.denorm(images.squeeze(1))
        else:
            return images.squeeze(1)

    def decode_soft(
        self,
        logits,
    ):
        # Actually, the effect of the selection of tau is minor
        soft_one_hot = F.softmax(logits*self.tau, dim=1)
        image_embeds = einsum('b n h w, n d -> b d h w',
                              soft_one_hot, self._vq_vae._embedding)
        images = self.decoder(image_embeds)

        if self.train_objective == 'classification':
            images = images.argmax(dim=1, keepdim=True)
        if self.use_sigmoid:
            return images.sigmoid().squeeze(1)
        elif self.use_norm:
            return self.denorm(images.squeeze(1))
        else:
            return images.squeeze(1)

    def forward(
        self,
        img,
        use_norm=None
    ):
        device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        assert img.shape[-1] == image_size and img.shape[
            -2] == image_size, f'input must have the correct image size {image_size}'
        label_img = img.clone()
        img = img.float()

        use_norm = self.use_norm if use_norm is None else use_norm
        if use_norm:
            img = self.norm(img)

        logits = self.encoder(img)
        vq_loss, quantized, perplexity, _, code_indices = self._vq_vae(logits)
        out = self.decoder(quantized)

        # reconstruction loss
        if self.train_objective == 'classification':
            recon_loss = self.loss_fn(out, label_img[:, 0, :, :].long())
        else:
            recon_loss = self.loss_fn(out, img)

        total_loss = recon_loss + vq_loss

        if self.train_objective == 'classification':
            out = out.argmax(dim=1, keepdim=True)

        if self.use_sigmoid:
            out = out.sigmoid()
        elif use_norm:
            out = self.denorm(out)
        return total_loss, recon_loss, vq_loss, out


#=============================================================================#
#                         ait/code/model/optimizer.py                         #
#=============================================================================#

import warnings
import json

import torch
from torch.nn import GroupNorm, LayerNorm

from mmcv.utils import _BatchNorm, _InstanceNorm, build_from_cfg, is_list_of
from mmcv.utils.ext_loader import check_ops_exist
from mmcv.runner import OPTIMIZER_BUILDERS, OPTIMIZERS, DefaultOptimizerConstructor
from mmcv.runner import get_dist_info


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.patch_embed"):
        return 0
    elif var_name.startswith("backbone.blocks"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1


def get_num_layer_for_swin(var_name, num_max_layer, layers_per_stage):
    if var_name in ("backbone.cls_token", "backbone.mask_token",
                    "backbone.pos_embed", "backbone.absolute_pos_embed"):
        return 0
    elif var_name.startswith("backbone.patch_embed"):
        return 0
    elif var_name.startswith("backbone.layers"):
        if var_name.split('.')[3] == "blocks":
            stage_id = int(var_name.split('.')[2])
            layer_id = int(var_name.split('.')[4]) \
                + sum(layers_per_stage[:stage_id])
            return layer_id + 1
        elif var_name.split('.')[3] == "downsample":
            stage_id = int(var_name.split('.')[2])
            layer_id = sum(layers_per_stage[:stage_id+1])
            return layer_id
    else:
        return num_max_layer - 1


@OPTIMIZER_BUILDERS.register_module()
class CustomOptimizerConstructor:
    """Default constructor for optimizers.

    By default each parameter share the same optimizer settings, and we
    provide an argument ``paramwise_cfg`` to specify parameter-wise settings.
    It is a dict and may contain the following fields:

    - ``custom_keys`` (dict): Specified parameters-wise settings by keys. If
      one of the keys in ``custom_keys`` is a substring of the name of one
      parameter, then the setting of the parameter will be specified by
      ``custom_keys[key]`` and other setting like ``bias_lr_mult`` etc. will
      be ignored. It should be noted that the aforementioned ``key`` is the
      longest key that is a substring of the name of the parameter. If there
      are multiple matched keys with the same length, then the key with lower
      alphabet order will be chosen.
      ``custom_keys[key]`` should be a dict and may contain fields ``lr_mult``
      and ``decay_mult``. See Example 2 below.
    - ``bias_lr_mult`` (float): It will be multiplied to the learning
      rate for all bias parameters (except for those in normalization
      layers and offset layers of DCN).
    - ``bias_decay_mult`` (float): It will be multiplied to the weight
      decay for all bias parameters (except for those in
      normalization layers, depthwise conv layers, offset layers of DCN).
    - ``norm_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of normalization
      layers.
    - ``dwconv_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of depthwise conv
      layers.
    - ``dcn_offset_lr_mult`` (float): It will be multiplied to the learning
      rate for parameters of offset layer in the deformable convs
      of a model.
    - ``bypass_duplicate`` (bool): If true, the duplicate parameters
      would not be added into optimizer. Default: False.

    Note:
        1. If the option ``dcn_offset_lr_mult`` is used, the constructor will
            override the effect of ``bias_lr_mult`` in the bias of offset
            layer. So be careful when using both ``bias_lr_mult`` and
            ``dcn_offset_lr_mult``. If you wish to apply both of them to the
            offset layer in deformable convs, set ``dcn_offset_lr_mult``
            to the original ``dcn_offset_lr_mult`` * ``bias_lr_mult``.
        2. If the option ``dcn_offset_lr_mult`` is used, the constructor will
            apply it to all the DCN layers in the model. So be carefull when
            the model contains multiple DCN layers in places other than
            backbone.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are

                - `type`: class name of the optimizer.

            Optional fields are

                - any arguments of the corresponding optimizer type, e.g.,
                  lr, weight_decay, momentum, etc.
        paramwise_cfg (dict, optional): Parameter-wise options.

    Example 1:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> paramwise_cfg = dict(norm_decay_mult=0.)
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)

    Example 2:
        >>> # assume model have attribute model.backbone and model.cls_head
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, weight_decay=0.95)
        >>> paramwise_cfg = dict(custom_keys={
                '.backbone': dict(lr_mult=0.1, decay_mult=0.9)})
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)
        >>> # Then the `lr` and `weight_decay` for model.backbone is
        >>> # (0.01 * 0.1, 0.95 * 0.9). `lr` and `weight_decay` for
        >>> # model.cls_head is (0.01, 0.95).
    """

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        if not isinstance(optimizer_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict',
                            f'but got {type(optimizer_cfg)}')
        self.optimizer_cfg = optimizer_cfg
        self.paramwise_cfg = {} if paramwise_cfg is None else paramwise_cfg
        self.base_lr = optimizer_cfg.get('lr', None)
        self.base_wd = optimizer_cfg.get('weight_decay', None)
        self._validate_cfg()

    def _validate_cfg(self):
        if not isinstance(self.paramwise_cfg, dict):
            raise TypeError('paramwise_cfg should be None or a dict, '
                            f'but got {type(self.paramwise_cfg)}')

        if 'custom_keys' in self.paramwise_cfg:
            if not isinstance(self.paramwise_cfg['custom_keys'], dict):
                raise TypeError(
                    'If specified, custom_keys must be a dict, '
                    f'but got {type(self.paramwise_cfg["custom_keys"])}')
            if self.base_wd is None:
                for key in self.paramwise_cfg['custom_keys']:
                    if 'decay_mult' in self.paramwise_cfg['custom_keys'][key]:
                        raise ValueError('base_wd should not be None')

        # get base lr and weight decay
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in self.paramwise_cfg
                or 'norm_decay_mult' in self.paramwise_cfg
                or 'dwconv_decay_mult' in self.paramwise_cfg):
            if self.base_wd is None:
                raise ValueError('base_wd should not be None')

    def _is_in(self, param_group, param_group_list):
        assert is_list_of(param_group_list, dict)
        param = set(param_group['params'])
        param_set = set()
        for group in param_group_list:
            param_set.update(set(group['params']))

        return not param.isdisjoint(param_set)

    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', 1.)
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', 1.)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', 1.)
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', 1.)
        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', False)
        dcn_offset_lr_mult = self.paramwise_cfg.get('dcn_offset_lr_mult', 1.)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module,
                             (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (
            isinstance(module, torch.nn.Conv2d)
            and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue
            if bypass_duplicate and self._is_in(param_group, params):
                warnings.warn(f'{prefix} is duplicate. It is skipped since '
                              f'bypass_duplicate={bypass_duplicate}')
                continue
            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in sorted_keys:
                if key in f'{prefix}.{name}':
                    is_custom = True
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    param_group['lr'] = self.base_lr * lr_mult
                    if lr_mult != 1.:
                        print(
                            f'==========> learning rate of {prefix}.{name}: {self.base_lr * lr_mult}')
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get('decay_mult', 1.)
                        param_group['weight_decay'] = self.base_wd * decay_mult
                        if decay_mult != 1.:
                            print(
                                f'==========> weight decay of {prefix}.{name}: {self.base_wd * decay_mult}')
                    # break

            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if name == 'bias' and not (is_norm or is_dcn_module):
                    param_group['lr'] = self.base_lr * bias_lr_mult

                if (prefix.find('conv_offset') != -1 and is_dcn_module
                        and isinstance(module, torch.nn.Conv2d)):
                    # deal with both dcn_offset's bias & weight
                    param_group['lr'] = self.base_lr * dcn_offset_lr_mult

                # apply weight decay policies
                if self.base_wd is not None:
                    # norm decay
                    if is_norm:
                        param_group[
                            'weight_decay'] = self.base_wd * norm_decay_mult
                    # depth-wise conv
                    elif is_dwconv:
                        param_group[
                            'weight_decay'] = self.base_wd * dwconv_decay_mult
                    # bias lr and decay
                    elif name == 'bias' and not is_dcn_module:
                        # TODO: current bias_decay_mult will have affect on DCN
                        param_group[
                            'weight_decay'] = self.base_wd * bias_decay_mult
            params.append(param_group)

        if check_ops_exist():
            from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
            is_dcn_module = isinstance(module,
                                       (DeformConv2d, ModulatedDeformConv2d))
        else:
            is_dcn_module = False
        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(
                params,
                child_mod,
                prefix=child_prefix,
                is_dcn_module=is_dcn_module)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = model.parameters()
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        # set param-wise lr and weight decay recursively
        params = []
        self.add_params(params, model)
        optimizer_cfg['params'] = params

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)


@OPTIMIZER_BUILDERS.register_module()
class LayerDecayOptimizerConstructor(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}
        print(self.paramwise_cfg)
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        print("Build LayerDecayOptimizerConstructor %f - %d" %
              (layer_decay_rate, num_layers))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in ('pos_embed', 'cls_token'):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            layer_id = get_num_layer_for_vit(name, num_layers)
            group_name = "layer_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * self.base_lr,
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"],
                    "lr_scale": parameter_groups[key]["lr_scale"],
                    "lr": parameter_groups[key]["lr"],
                    "weight_decay": parameter_groups[key]["weight_decay"],
                }
            print("Param groups = %s" % json.dumps(to_display, indent=2))

        # state_dict = module.state_dict()
        # for group_name in parameter_groups:
        #     group = parameter_groups[group_name]
        #     for name in group["param_names"]:
        #         group["params"].append(state_dict[name])
        params.extend(parameter_groups.values())


@OPTIMIZER_BUILDERS.register_module()
class SwinLayerDecayOptimizerConstructor(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}
        print(self.paramwise_cfg)
        layers_per_stage = self.paramwise_cfg.get('num_layers')
        no_decay_names = self.paramwise_cfg.get('no_decay_names', [])
        lr_mul = self.paramwise_cfg.get('lr_mul', {})
        for i in range(len(layers_per_stage) - 1):
            layers_per_stage[i] = layers_per_stage[i] + 1  # patch merging
        num_layers = sum(layers_per_stage) + 2  # 2: patch embed, head
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        print("Build SwinLayerDecayOptimizerConstructor %f - %d" %
              (layer_decay_rate, num_layers))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in ('absolute_pos_embed'):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

                for nd_name in no_decay_names:
                    if nd_name in name:
                        group_name = "no_decay"
                        this_weight_decay = 0.
                        break

            layer_id = get_num_layer_for_swin(
                name, num_layers, layers_per_stage)
            lr_mul_ratio = 1.
            for lrmul_name, ratio in lr_mul.items():
                if lrmul_name in name:
                    lr_mul_ratio = ratio
                    break

            group_name = "layer_%d_%s_%f" % (
                layer_id, group_name, lr_mul_ratio)

            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [],
                    "lr_scale": scale * lr_mul_ratio,
                    "group_name": group_name,
                    "lr": scale * self.base_lr,
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"],
                    "lr_scale": parameter_groups[key]["lr_scale"],
                    "lr": parameter_groups[key]["lr"],
                    "weight_decay": parameter_groups[key]["weight_decay"],
                }
            print("Param groups = %s" % json.dumps(to_display, indent=2))

        # state_dict = module.state_dict()
        # for group_name in parameter_groups:
        #     group = parameter_groups[group_name]
        #     for name in group["param_names"]:
        #         group["params"].append(state_dict[name])
        params.extend(parameter_groups.values())
