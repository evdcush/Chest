#=============================================================================#
#      ____       ______      __  __      _____                  ___          #
#     /\  _`\    /\__  _\    /\ \/\ \    /\  __`\              /'___`\        #
#     \ \ \/\ \  \/_/\ \/    \ \ `\\ \   \ \ \/\ \    __  __  /\_\ /\ \       #
#      \ \ \ \ \    \ \ \     \ \ , ` \   \ \ \ \ \  /\ \/\ \ \/_/// /__      #
#       \ \ \_\ \    \_\ \__   \ \ \`\ \   \ \ \_\ \ \ \ \_/ |   // /_\ \     #
#        \ \____/    /\_____\   \ \_\ \_\   \ \_____\ \ \___/   /\______/     #
#         \/___/     \/_____/    \/_/\/_/    \/_____/  \/__/    \/_____/      #
#                                                                             #
#=============================================================================#
# https://github.com/facebookresearch/dinov2
# Ref: da4b382
#
#
# dinov2
# ├── dinov2/
# │   ├── data/
# │   │   ├── datasets/
# │   │   │   ├── decoders.py
# │   │   │   ├── extended.py
# │   │   │   ├── image_net_22k.py
# │   │   │   └── image_net.py
# │   │   ├── adapters.py
# │   │   ├── augmentations.py
# │   │   ├── collate.py
# │   │   ├── loaders.py
# │   │   ├── masking.py
# │   │   ├── samplers.py
# │   │   └── transforms.py
# │   ├── eval/
# │   │   ├── depth/
# │   │   │   ├── models/
# │   │   │   │   ├── backbones/
# │   │   │   │   │   └── vision_transformer.py
# │   │   │   │   ├── decode_heads/
# │   │   │   │   │   ├── decode_head.py
# │   │   │   │   │   ├── dpt_head.py
# │   │   │   │   │   └── linear_head.py
# │   │   │   │   ├── depther/
# │   │   │   │   │   ├── base.py
# │   │   │   │   │   └── encoder_decoder.py
# │   │   │   │   ├── losses/
# │   │   │   │   │   ├── gradientloss.py
# │   │   │   │   │   └── sigloss.py
# │   │   │   │   └── builder.py
# │   │   │   └── ops/
# │   │   │       └── wrappers.py
# │   │   ├── segmentation/
# │   │   │   ├── hooks/
# │   │   │   │   └── optimizer.py
# │   │   │   ├── models/
# │   │   │   │   ├── backbones/
# │   │   │   │   │   └── vision_transformer.py
# │   │   │   │   └── decode_heads/
# │   │   │   │       └── linear_head.py
# │   │   │   └── utils/
# │   │   │       └── colormaps.py
# │   │   ├── segmentation_m2f/
# │   │   │   ├── core/
# │   │   │   │   ├── anchor/
# │   │   │   │   │   ├── builder.py
# │   │   │   │   │   └── point_generator.py
# │   │   │   │   ├── box/
# │   │   │   │   │   ├── samplers/
# │   │   │   │   │   │   ├── base_sampler.py
# │   │   │   │   │   │   ├── mask_pseudo_sampler.py
# │   │   │   │   │   │   ├── mask_sampling_result.py
# │   │   │   │   │   │   └── sampling_result.py
# │   │   │   │   │   └── builder.py
# │   │   │   │   └── utils/
# │   │   │   │       ├── dist_utils.py
# │   │   │   │       └── misc.py
# │   │   │   ├── models/
# │   │   │   │   ├── backbones/
# │   │   │   │   │   ├── adapter_modules.py
# │   │   │   │   │   ├── drop_path.py
# │   │   │   │   │   ├── vit_adapter.py
# │   │   │   │   │   └── vit.py
# │   │   │   │   ├── decode_heads/
# │   │   │   │   │   └── mask2former_head.py
# │   │   │   │   ├── losses/
# │   │   │   │   │   ├── cross_entropy_loss.py
# │   │   │   │   │   ├── dice_loss.py
# │   │   │   │   │   └── match_costs.py
# │   │   │   │   ├── plugins/
# │   │   │   │   │   └── msdeformattn_pixel_decoder.py
# │   │   │   │   ├── segmentors/
# │   │   │   │   │   └── encoder_decoder_mask2former.py
# │   │   │   │   ├── utils/
# │   │   │   │   │   ├── assigner.py
# │   │   │   │   │   ├── point_sample.py
# │   │   │   │   │   ├── positional_encoding.py
# │   │   │   │   │   └── transformer.py
# │   │   │   │   └── builder.py
# │   │   │   └── ops/
# │   │   │       └── modules/
# │   │   │           └── ms_deform_attn.py
# │   │   ├── knn.py
# │   │   ├── linear.py
# │   │   ├── log_regression.py
# │   │   ├── metrics.py
# │   │   ├── setup.py
# │   │   └── utils.py
# │   ├── hub/
# │   │   ├── depth/
# │   │   │   ├── decode_heads.py
# │   │   │   ├── encoder_decoder.py
# │   │   │   └── ops.py
# │   │   ├── backbones.py
# │   │   ├── classifiers.py
# │   │   ├── depthers.py
# │   │   └── utils.py
# │   ├── layers/
# │   │   ├── attention.py
# │   │   ├── block.py
# │   │   ├── dino_head.py
# │   │   ├── drop_path.py
# │   │   ├── layer_scale.py
# │   │   ├── mlp.py
# │   │   ├── patch_embed.py
# │   │   └── swiglu_ffn.py
# │   ├── loss/
# │   │   ├── dino_clstoken_loss.py
# │   │   ├── ibot_patch_loss.py
# │   │   └── koleo_loss.py
# │   ├── models/
# │   │   └── vision_transformer.py
# │   ├── train/
# │   │   ├── ssl_meta_arch.py
# │   │   └── train.py
# │   └── utils/
# │       ├── cluster.py
# │       ├── config.py
# │       ├── dtype.py
# │       ├── param_groups.py
# │       └── utils.py
# └── notebooks/
#     ├── depth_estimation.ipynb
#     └── semantic_segmentation.ipynb

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
#                     dinov2/models/vision_transformer.py                     #
#=============================================================================#

#$#>START: dinov2/dinov2/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, \
NestedTensorBlock as Block

logger = logging.getLogger("dinov2")


def named_apply(fn: Callable,
                module: nn.Module,
                name="",
                depth_first=True,
                include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn,
                    module=child_module,
                    name=child_name,
                    depth_first=depth_first,
                    include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):

    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform(bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units
                for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called
                "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when
                interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when
                interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_chans=in_chans,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (nn.Parameter(
            torch.zeros(1, num_register_tokens, embed_dim))
                                if num_register_tokens else None)

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
                   ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            ) for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk
                # the block list
                chunked_blocks.append([nn.Identity()] * i +
                                      blocks_list[i:i + chunksize])
            self.blocks = nn.ModuleList(
                [BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the
        # interpolation see discussion at
        # https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N),
                                    dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed),
                         dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1),
                            self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [
            self.prepare_tokens_with_masks(x, masks)
            for x, masks in zip(x_list, masks_list)
        ]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append({
                "x_norm_clstoken":
                x_norm[:, 0],
                "x_norm_regtokens":
                x_norm[:, 1:self.num_register_tokens + 1],
                "x_norm_patchtokens":
                x_norm[:, self.num_register_tokens + 1:],
                "x_prenorm":
                x,
                "masks":
                masks,
            })
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1:self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len -
                               n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len -
                               n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size,
                            -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per
    head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


#$#>END: dinov2/dinov2/models/vision_transformer.py

#------------------------------------------------------------------------------

#$#>START: dinov2/dinov2/models/__init__.py

import logging

from . import vision_transformer as vits

logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student,
                       only_teacher=only_teacher,
                       img_size=cfg.crops.global_crops_size)


#$#>END: dinov2/dinov2/models/__init__.py

#=============================================================================#
#                                                                             #
#              ██       █████  ██    ██ ███████ ██████  ███████               #
#              ██      ██   ██  ██  ██  ██      ██   ██ ██                    #
#              ██      ███████   ████   █████   ██████  ███████               #
#              ██      ██   ██    ██    ██      ██   ██      ██               #
#              ███████ ██   ██    ██    ███████ ██   ██ ███████               #
#                                                                             #
#=============================================================================#

## imported in model::
#from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention,
#NestedTensorBlock as Block

#=============================================================================#
#                          dinov2/layers/dino_head.py                         #
#=============================================================================#

#$#>START: dinov2/dinov2/layers/dino_head.py

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


class DINOHead(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers,
                              in_dim,
                              bottleneck_dim,
                              hidden_dim=hidden_dim,
                              use_bn=use_bn,
                              bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


def _build_mlp(nlayers,
               in_dim,
               bottleneck_dim,
               hidden_dim=None,
               use_bn=False,
               bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)


#$#>END: dinov2/dinov2/layers/dino_head.py

#=============================================================================#
#                            dinov2/layers/block.py                           #
#=============================================================================#
## CONTENTS:
'''
CLASSES:
  - Block(nn.Module)
    - NestedTensorBlock(Block)

FUNCTIONS:
  - drop_add_residual_stochastic_depth
  - get_branges_scales
  - add_residual
  - get_attn_bias_and_cat
  - drop_add_residual_stochastic_depth_list
'''

#$#>START: dinov2/dinov2/layers/block.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

logger = logging.getLogger("dinov2")

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Block)")
    else:
        warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (Block)")


class Block(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:

        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate
            # larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat,
                                      0,
                                      brange,
                                      residual.to(dtype=x.dtype),
                                      alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x,
                 brange,
                 residual,
                 residual_scale_factor,
                 scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat,
                                          0,
                                          brange,
                                          residual.to(dtype=x.dtype),
                                          alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(x,
                                           brange,
                                           residual.to(dtype=x.dtype),
                                           scaling=scaling_vector,
                                           alpha=residual_scale_factor)
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and
    provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges
                   ] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list],
                                       branges).view(1, -1,
                                                     x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [
        get_branges_scales(x, sample_drop_ratio=sample_drop_ratio)
        for x in x_list
    ]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(
        x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(
            x_list, branges, residual_list, residual_scale_factors):
        outputs.append(
            add_residual(x, brange, residual, residual_scale_factor,
                         scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):

    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(
                    self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(
                    self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError(
                    "xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError


#$#>END: dinov2/dinov2/layers/block.py

#=============================================================================#
#                          dinov2/layers/attention.py                         #
#=============================================================================#

#$#>START: dinov2/dinov2/layers/attention.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn

logger = logging.getLogger("dinov2")

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError(
                    "xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#$#>END: dinov2/dinov2/layers/attention.py

#=============================================================================#
#                          dinov2/layers/drop_path.py                         #
#=============================================================================#

#$#>START: dinov2/dinov2/layers/drop_path.py

from torch import nn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path
    of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


#$#>END: dinov2/dinov2/layers/drop_path.py

#=============================================================================#
#                         dinov2/layers/layer_scale.py                        #
#=============================================================================#

#$#>START: dinov2/dinov2/layers/layer_scale.py

from typing import Union

import torch
from torch import Tensor
from torch import nn


class LayerScale(nn.Module):

    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


#$#>END: dinov2/dinov2/layers/layer_scale.py

#=============================================================================#
#                             dinov2/layers/mlp.py                            #
#=============================================================================#

#$#>START: dinov2/dinov2/layers/mlp.py

from typing import Callable, Optional

from torch import Tensor, nn


class Mlp(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


#$#>END: dinov2/dinov2/layers/mlp.py

#=============================================================================#
#                         dinov2/layers/patch_embed.py                        #
#=============================================================================#

#$#>START: dinov2/dinov2/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_HW,
                              stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, \
        f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, \
        f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (
            self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


#$#>END: dinov2/dinov2/layers/patch_embed.py

#=============================================================================#
#                         dinov2/layers/swiglu_ffn.py                         #
#=============================================================================#

#$#>START: dinov2/dinov2/layers/swiglu_ffn.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from typing import Callable, Optional
import warnings

from torch import Tensor, nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import SwiGLU

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (SwiGLU)")
    else:
        warnings.warn("xFormers is disabled (SwiGLU)")
        raise ImportError
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (SwiGLU)")


class SwiGLUFFNFused(SwiGLU):

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


#$#>END: dinov2/dinov2/layers/swiglu_ffn.py

#=============================================================================#
#                                                                             #
#                          ██   ██ ██    ██ ██████                            #
#                          ██   ██ ██    ██ ██   ██                           #
#                          ███████ ██    ██ ██████                            #
#                          ██   ██ ██    ██ ██   ██                           #
#                          ██   ██  ██████  ██████                            #
#                                                                             #
#=============================================================================#

#=============================================================================#
#                          dinov2/hub/classifiers.py                          #
#=============================================================================#

#$#>START: dinov2/dinov2/hub/classifiers.py

from enum import Enum
from typing import Union

import torch
import torch.nn as nn

from .backbones import _make_dinov2_model
from .utils import _DINOV2_BASE_URL, _make_dinov2_model_name


class Weights(Enum):
    IMAGENET1K = "IMAGENET1K"


def _make_dinov2_linear_classification_head(
    *,
    arch_name: str = "vit_large",
    patch_size: int = 14,
    embed_dim: int = 1024,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    num_register_tokens: int = 0,
    **kwargs,
):
    if layers not in (1, 4):
        raise AssertionError(f"Unsupported number of layers: {layers}")
    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    linear_head = nn.Linear((1 + layers) * embed_dim, 1_000)

    if pretrained:
        model_base_name = _make_dinov2_model_name(arch_name, patch_size)
        model_full_name = _make_dinov2_model_name(arch_name, patch_size,
                                                  num_register_tokens)
        layers_str = str(layers) if layers == 4 else ""
        url = _DINOV2_BASE_URL + \
        f"/{model_base_name}/{model_full_name}_linear{layers_str}_head.pth"
        state_dict = torch.hub.load_state_dict_from_url(url,
                                                        map_location="cpu")
        linear_head.load_state_dict(state_dict, strict=True)

    return linear_head


class _LinearClassifierWrapper(nn.Module):

    def __init__(self,
                 *,
                 backbone: nn.Module,
                 linear_head: nn.Module,
                 layers: int = 4):
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head
        self.layers = layers

    def forward(self, x):
        if self.layers == 1:
            x = self.backbone.forward_features(x)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            # fmt: off
            linear_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)
            # fmt: on
        elif self.layers == 4:
            x = self.backbone.get_intermediate_layers(x,
                                                      n=4,
                                                      return_class_token=True)
            # fmt: off
            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1),
            ], dim=1)
            # fmt: on
        else:
            assert False, f"Unsupported number of layers: {self.layers}"
        return self.linear_head(linear_input)


def _make_dinov2_linear_classifier(
    *,
    arch_name: str = "vit_large",
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    **kwargs,
):
    backbone = _make_dinov2_model(
        arch_name=arch_name,
        pretrained=pretrained,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
        **kwargs,
    )

    embed_dim = backbone.embed_dim
    patch_size = backbone.patch_size
    linear_head = _make_dinov2_linear_classification_head(
        arch_name=arch_name,
        patch_size=patch_size,
        embed_dim=embed_dim,
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=num_register_tokens,
    )

    return _LinearClassifierWrapper(backbone=backbone,
                                    linear_head=linear_head,
                                    layers=layers)


def dinov2_vits14_lc(
    *,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    **kwargs,
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-S/14
    backbone (optionally) pretrained on the LVD-142M dataset and trained
    on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_small",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )


def dinov2_vitb14_lc(
    *,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    **kwargs,
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-B/14
    backbone (optionally) pretrained on the LVD-142M dataset and trained on
    ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_base",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )


def dinov2_vitl14_lc(
    *,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    **kwargs,
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-L/14 backbone
    (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_large",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )


def dinov2_vitg14_lc(
    *,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    **kwargs,
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-g/14 backbone
    (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_giant2",
        layers=layers,
        ffn_layer="swiglufused",
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )


def dinov2_vits14_reg_lc(*,
                         layers: int = 4,
                         pretrained: bool = True,
                         weights: Union[Weights, str] = Weights.IMAGENET1K,
                         **kwargs):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-S/14 backbone
    with registers (optionally) pretrained on the LVD-142M dataset and
    trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_small",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitb14_reg_lc(*,
                         layers: int = 4,
                         pretrained: bool = True,
                         weights: Union[Weights, str] = Weights.IMAGENET1K,
                         **kwargs):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-B/14 backbone
    with registers (optionally) pretrained on the LVD-142M dataset and
    trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_base",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitl14_reg_lc(*,
                         layers: int = 4,
                         pretrained: bool = True,
                         weights: Union[Weights, str] = Weights.IMAGENET1K,
                         **kwargs):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-L/14 backbone
    with registers (optionally) pretrained on the LVD-142M dataset and
    trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_large",
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitg14_reg_lc(*,
                         layers: int = 4,
                         pretrained: bool = True,
                         weights: Union[Weights, str] = Weights.IMAGENET1K,
                         **kwargs):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-g/14 backbone
    with registers (optionally) pretrained on the LVD-142M dataset and
    trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_giant2",
        layers=layers,
        ffn_layer="swiglufused",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


#$#>END: dinov2/dinov2/hub/classifiers.py

#=============================================================================#
#                           dinov2/hub/backbones.py                           #
#=============================================================================#

#$#>START: dinov2/dinov2/hub/backbones.py

from enum import Enum
from typing import Union

import torch

from .utils import _DINOV2_BASE_URL, _make_dinov2_model_name


class Weights(Enum):
    LVD142M = "LVD142M"


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD142M,
    **kwargs,
):
    from ..models import vision_transformer as vits

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        model_full_name = _make_dinov2_model_name(arch_name, patch_size,
                                                  num_register_tokens)
        url = _DINOV2_BASE_URL + \
        f"/{model_base_name}/{model_full_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url,
                                                        map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    return model


def dinov2_vits14(*,
                  pretrained: bool = True,
                  weights: Union[Weights, str] = Weights.LVD142M,
                  **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_small",
                              pretrained=pretrained,
                              weights=weights,
                              **kwargs)


def dinov2_vitb14(*,
                  pretrained: bool = True,
                  weights: Union[Weights, str] = Weights.LVD142M,
                  **kwargs):
    """
    DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_base",
                              pretrained=pretrained,
                              weights=weights,
                              **kwargs)


def dinov2_vitl14(*,
                  pretrained: bool = True,
                  weights: Union[Weights, str] = Weights.LVD142M,
                  **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_large",
                              pretrained=pretrained,
                              weights=weights,
                              **kwargs)


def dinov2_vitg14(*,
                  pretrained: bool = True,
                  weights: Union[Weights, str] = Weights.LVD142M,
                  **kwargs):
    """
    DINOv2 ViT-g/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        **kwargs,
    )


def dinov2_vits14_reg(*,
                      pretrained: bool = True,
                      weights: Union[Weights, str] = Weights.LVD142M,
                      **kwargs):
    """
    DINOv2 ViT-S/14 model with registers (optionally) pretrained on the
    LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_small",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitb14_reg(*,
                      pretrained: bool = True,
                      weights: Union[Weights, str] = Weights.LVD142M,
                      **kwargs):
    """
    DINOv2 ViT-B/14 model with registers (optionally) pretrained on the
    LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_base",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitl14_reg(*,
                      pretrained: bool = True,
                      weights: Union[Weights, str] = Weights.LVD142M,
                      **kwargs):
    """
    DINOv2 ViT-L/14 model with registers (optionally) pretrained on the
    LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_large",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitg14_reg(*,
                      pretrained: bool = True,
                      weights: Union[Weights, str] = Weights.LVD142M,
                      **kwargs):
    """
    DINOv2 ViT-g/14 model with registers (optionally) pretrained on the
    LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


#$#>END: dinov2/dinov2/hub/backbones.py
