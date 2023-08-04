#=============================================================================#
#                                                                             #
#                           ██████ ███████  ██████                            #
#                          ██      ██      ██                                 #
#                          ██      █████   ██   ███                           #
#                          ██      ██      ██    ██                           #
#                           ██████ ██       ██████                            #
#                                                                             #
#=============================================================================#

#=============================================================================#
#             big_vision/configs/proj/uvim/vqvae_coco_panoptic.py             #
#=============================================================================#
#====  STAGE I

"""A config for training a UViM stage I model for the panoptic task.

This config is expected to reproduce the paper's result and achieve
approximately 75.7 PQ points on the COCO holdout data.

We also provide a low-resource variant of this config, which can be enabled
by adding `:singlehost` postfix to the config name. This one is expected to
achieve 67.8 PQ points on the COCO holdout data.
"""

import itertools
import big_vision.configs.common as bvcc
import ml_collections as mlc


def get_config(arg='res=512,patch_size=16'):
    """Config for training label compression on COCO-panoptic."""
    arg = bvcc.parse_arg(arg,
                         res=512,
                         patch_size=16,
                         runlocal=False,
                         singlehost=False)
    config = mlc.ConfigDict()

    config.task = 'proj.uvim.panoptic_task'

    config.input = {}
    config.input.data = dict(name='coco/2017_panoptic', split='train[4096:]')

    config.input.batch_size = 1024
    config.input.shuffle_buffer_size = 25_000

    config.total_epochs = 1000

    config.input.pp = (
        f'decode|coco_panoptic|concat(["semantics","instances"], "labels")|'
        f'randu("fliplr")|det_fliplr(key="image")|det_fliplr(key="labels")|'
        f'inception_box|crop_box(key="image")|crop_box(key="labels")|'
        f'resize({arg.res})|resize({arg.res},key="labels",method="nearest")|'
        f'value_range(-1, 1)|make_canonical|keep("image","labels")')
    pp_eval = (
        f'decode|coco_panoptic|concat(["semantics","instances"], "labels")|'
        f'resize({arg.res})|resize({arg.res},key="labels",method="nearest")|'
        f'value_range(-1, 1)|make_canonical|keep("image","labels")')

    config.log_training_steps = 50
    config.ckpt_steps = 1000
    config.keep_ckpt_steps = 20_000

    # Model section
    config.model_name = 'proj.uvim.vit'
    config.model = mlc.ConfigDict()
    config.model.input_size = (arg.res, arg.res)
    config.model.patch_size = (arg.patch_size, arg.patch_size)
    config.model.code_len = 256
    config.model.width = 768
    config.model.enc_depth = 6
    config.model.dec_depth = 12
    config.model.mlp_dim = 3072
    config.model.num_heads = 12
    config.model.dict_size = 4096  # Number of words in dict.
    config.model.codeword_dim = 768
    config.model.dict_momentum = 0.995  # Momentum for dict. learning.
    config.model.with_encoder_ctx = True
    config.model.with_decoder_ctx = True
    config.model.code_dropout = 'random'
    config.model.bottleneck_resize = True
    config.model.inputs = {
        'semantics': (133 + 1, arg.patch_size**2),  # +1 for void label
        'instances':
        (100, arg.patch_size**2),  # COCO: actually 98 train/78 validation.
    }
    config.model.outputs = config.model.inputs

    # VQVAE-specific params.
    config.freeze_dict = False  # Will freeze a dict. inside VQ-VAE model.
    config.w_commitment = 0.0

    # Optimizer section
    config.optax_name = 'big_vision.scale_by_adafactor'
    config.optax = dict(beta2_cap=0.95)

    config.lr = 4e-4
    config.wd = 4e-5
    config.schedule = dict(decay_type='cosine', warmup_steps=4_000)
    config.grad_clip_norm = 1.0

    # Evaluation section
    config.evals = {}
    config.evals.val = mlc.ConfigDict()
    config.evals.val.type = 'proj.uvim.compute_mean'
    config.evals.val.pred = 'validation'
    config.evals.val.data = {**config.input.data}
    config.evals.val.data.split = 'train[:4096]'
    config.evals.val.pp_fn = pp_eval
    config.evals.val.log_steps = 250

    base = {
        'type': 'proj.uvim.coco_panoptic',
        'pp_fn': pp_eval.replace('decode|', ''),
        'log_steps': 10_000,
        # Filters objects that occupy less than 0.03^2 fraction of all pixels.
        # 'predict_kwargs': {'min_fraction': 0.03 ** 2},
    }
    config.evals.coco_panoptic_train = dict(**base, split='train[4096:8192]')
    config.evals.coco_panoptic_holdout = dict(**base, split='train[:4096]')
    config.evals.coco_panoptic = dict(**base, split='validation')

    # config.evals.save_pred = dict(type='proj.uvim.save_predictions')
    # config.evals.save_pred.pp = pp_eval.replace('decode|', '')
    # config.evals.save_pred.log_steps = 100_000
    # config.evals.save_pred.dataset = config.dataset
    # config.evals.save_pred.split = 'validation[:1024]'
    # config.evals.save_pred.outfile = 'inference.npz'

    config.seed = 0

    if arg.singlehost:
        config.input.batch_size = 128
        config.num_epochs = 100
    elif arg.runlocal:
        config.input.batch_size = 16
        config.input.shuffle_buffer_size = 10
        config.log_training_steps = 5
        config.model.enc_depth = 1
        config.model.dec_depth = 1
        config.evals.val.data.split = 'validation[:16]'
        config.evals.val.log_steps = 20

    return config


#=============================================================================#
#        big_vision/configs/proj/uvim/train_coco_panoptic_pretrained.py       #
#=============================================================================#
#====  STAGE II

"""A config for training a UViM stage II model for the panoptic task.

This config is expected to reproduce the paper's result and achieve
approximately 43.7 PQ points on the COCO holdout data.

We also provide a low-resource variant of this config, which can be enabled
by adding `:singlehost` postfix to the config name. This one is expected to
achieve 39.4 PQ points on the COCO holdout data.
"""

import big_vision.configs.common as bvcc
from ml_collections import ConfigDict

VTT_MODELS = {
    'base': dict(num_layers=12, num_heads=12, mlp_dim=3072, emb_dim=768),
    'large': dict(num_layers=24, num_heads=16, mlp_dim=4096, emb_dim=1024),
}

VQVAE_MODELS = {
    'base': dict(enc_depth=6,
                 dec_depth=12,
                 num_heads=12,
                 mlp_dim=3072,
                 width=768),
}

RES = 512
PATCH_SIZE = 16
LABEL_RES = 512
LABEL_PATCH_SIZE = 16


def get_config(arg=''):
    """Config for training."""
    arg = bvcc.parse_arg(arg, runlocal=False, singlehost=False)
    config = ConfigDict()

    config.input = {}
    config.input.pp = (
        f'decode|coco_panoptic|concat(["semantics","instances"], "labels")|'
        f'randu("fliplr")|det_fliplr(key="image")|det_fliplr(key="labels")|'
        f'inception_box|crop_box(key="image")|crop_box(key="labels")|'
        f'resize({LABEL_RES}, inkey="image", outkey="image_ctx")|'
        f'resize({RES})|resize({LABEL_RES},key="labels",method="nearest")|'
        f'value_range(-1, 1, key="image_ctx")|'
        f'value_range(-1, 1)|make_canonical|keep("image","image_ctx","labels")'
    )
    pp_eval = (
        f'decode|coco_panoptic|concat(["semantics","instances"], "labels")|'
        f'resize({LABEL_RES}, inkey="image", outkey="image_ctx")|'
        f'resize({RES})|resize({LABEL_RES},key="labels",method="nearest")|'
        f'value_range(-1, 1, key="image_ctx")|'
        f'value_range(-1, 1)|make_canonical|keep("image","image_ctx","labels")'
    )
    pp_predict = (
        f'resize({LABEL_RES}, inkey="image", '
        f'outkey="image_ctx")|resize({RES})|'
        f'value_range(-1, 1, key="image_ctx")|value_range(-1, 1)|'
        f'keep("image","image_ctx","image/id")'  # image/id used for rng seeds.
    )

    config.input.data = dict(name='coco/2017_panoptic', split='train[4096:]')
    config.input.batch_size = 512
    config.input.shuffle_buffer_size = 50_000

    config.total_epochs = 200

    config.log_training_steps = 50
    config.ckpt_steps = 1000
    config.keep_ckpt_steps = 5000
    config.prefetch_to_device = 2
    config.seed = 0

    # Optimizer section
    config.optax_name = 'big_vision.scale_by_adafactor'
    config.optax = dict(beta2_cap=0.95)

    config.lr = 0.001
    config.wd = 0.000001
    config.lr_mults = [('pos_embedding_encoder.*', 0.1),
                       ('EmbedPatches.*', 0.1), ('encoder.*', 0.1),
                       ('decoder.*', 1.0)]
    config.schedule = dict(decay_type='cosine', warmup_steps=4_000)

    # Oracle section
    config.oracle = ConfigDict()
    config.oracle.task = 'proj.uvim.panoptic_task'
    config.oracle.model_init = 'gs://big_vision/uvim/panoptic_stageI_params.npz'
    config.oracle.model_name = 'proj.uvim.vit'
    config.oracle.model = ConfigDict(VQVAE_MODELS['base'])
    config.oracle.model.input_size = (LABEL_RES, LABEL_RES)
    config.oracle.model.patch_size = (LABEL_PATCH_SIZE, LABEL_PATCH_SIZE)
    config.oracle.model.code_len = 256
    config.oracle.model.dict_size = 4096
    config.oracle.model.codeword_dim = 768
    config.oracle.model.with_encoder_ctx = True
    config.oracle.model.with_decoder_ctx = True
    config.oracle.model.code_dropout = 'random'
    config.oracle.model.bottleneck_resize = True
    config.oracle.model.inputs = {
        'semantics': (133 + 1, LABEL_PATCH_SIZE**2),  # +1 for void label
        'instances':
        (100, LABEL_PATCH_SIZE**2),  # COCO: actually 98 train/78 validation.
    }
    config.oracle.model.outputs = config.oracle.model.inputs

    # Model section
    config.model_name = 'proj.uvim.vtt'
    # config.model_init = {'encoder': 'howto-i21k-B/8'}
    config.model_init = {'encoder': 'howto-i21k-L/16'}
    config.model = ConfigDict(VTT_MODELS['large'])
    config.model.patches = ConfigDict({'size': (PATCH_SIZE, PATCH_SIZE)})
    config.model.vocab_size = config.oracle.model.get_ref('dict_size') + 1
    config.model.posemb_type = 'learn'
    config.model.input_size = (RES, RES)
    config.model.seq_len = config.oracle.model.get_ref('code_len')

    # Evaluation section
    config.evals = {}
    config.evals.val = ConfigDict()
    config.evals.val.type = 'proj.uvim.compute_mean'
    config.evals.val.pred = 'validation'
    config.evals.val.data = dict(name=config.input.data.name,
                                 split='train[:4096]')
    config.evals.val.pp_fn = pp_eval
    config.evals.val.log_steps = 1000

    base = {
        'type': 'proj.uvim.coco_panoptic',
        'pp_fn': pp_predict,
        'log_steps': 10_000,
        # Filters objects that occupy less than 0.03^2 fraction of all pixels.
        # 'predict_kwargs': {'min_fraction': 0.03 ** 2},
    }
    config.evals.coco_panoptic_train = dict(**base, split='train[4096:8192]')
    config.evals.coco_panoptic_holdout = dict(**base, split='train[:4096]')
    config.evals.coco_panoptic = dict(**base, split='validation')

    # config.evals.save_pred = dict(type='proj.uvim.save_predictions')
    # config.evals.save_pred.pp = pp_eval.replace('decode|', '')
    # config.evals.save_pred.log_steps = 100_000
    # config.evals.save_pred.dataset = config.dataset
    # config.evals.save_pred.split = 'validation[:1024]'
    # config.evals.save_pred.outfile = 'inference.npz'

    if arg.singlehost:
        config.input.batch_size = 32
        config.num_epochs = 50
    elif arg.runlocal:
        config.input.batch_size = 4
        config.input.shuffle_buffer_size = 10
        config.evals.val.data.split = 'train[:16]'
    return config


#=============================================================================#
#                                                                             #
#             ███    ███  ██████  ██████  ███████ ██      ███████             #
#             ████  ████ ██    ██ ██   ██ ██      ██      ██                  #
#             ██ ████ ██ ██    ██ ██   ██ █████   ██      ███████             #
#             ██  ██  ██ ██    ██ ██   ██ ██      ██           ██             #
#             ██      ██  ██████  ██████  ███████ ███████ ███████             #
#                                                                             #
#=============================================================================#

#=============================================================================#
#                         big_vision/models/common.py                         #
#=============================================================================#

"""Utilities shared across models."""

from absl import logging
import big_vision.utils as u
import flax.linen as nn
import jax
import jax.numpy as jnp


def merge_params(loaded, inited, dont_load=()):
    """Makes `loaded` pytree match `init`, warning or failing on mismatch.

    Args:
        loaded: pytree of parameters, typically loaded from a checkpoint.
        inited: pytree of parameter, typically coming from model init.
        dont_load: List of regexes for parameters which shall not be taken
            from `loaded`, either because they should remain at their init
            value, or because they are missing on either side.

    Returns:
        If successful, a new pytree which matches the structure of `init`
        but contains values from `loaded`, except for `dont_load`.
        If structures don't match and mismatches are not covered by regexes in
        `dont_load` argument, then raises an exception with more information.
  """
    if inited is None:  # A useful shortcut for example for colabs.
        return loaded

    dont_load = u.check_and_compile_patterns(dont_load)

    def should_merge(name):
        return not any(pattern.fullmatch(name) for pattern in dont_load)

    loaded_flat, _ = u.tree_flatten_with_names(loaded)
    inited_flat, _ = u.tree_flatten_with_names(inited)
    loaded_flat = {k: v for k, v in loaded_flat}
    inited_flat = {k: v for k, v in inited_flat}

    # Let's first build the pytree from all common keys.
    merged = {}
    for name, init_val in inited_flat.items():
        # param is present in both. Load or ignore it!
        if name in loaded_flat and should_merge(name):
            merged[name] = loaded_flat[name]
        else:
            logging.info("Ignoring checkpoint and using init value for %s",
                         name)
            merged[name] = init_val

    def pp(title, names, indent="  "):  # Just pretty-printing
        if names:
            return f"{title}:\n" + "\n".join(f"{indent}{k}"
                                             for k in sorted(names))
        else:
            return ""

    # Now, if there are keys that only exist in inited or loaded, be helpful:
    not_in_loaded = inited_flat.keys() - loaded_flat.keys()
    not_in_inited = loaded_flat.keys() - inited_flat.keys()
    logging.info(pp("Parameters in model but not in checkpoint",
                    not_in_loaded))
    logging.info(pp("Parameters in checkpoint but not in model",
                    not_in_inited))

    # And now see if any of them are not explicitly ignored => an error
    not_in_loaded = {k for k in not_in_loaded if should_merge(k)}
    not_in_inited = {k for k in not_in_inited if should_merge(k)}

    if not_in_loaded or not_in_inited:
        raise ValueError(
            pp("Params in checkpoint", loaded_flat.keys()) + "\n" +
            pp("Params in model (code)", inited_flat.keys()) + "\n" +
            pp("Params in model (code) but not in checkpoint and not"
               "`dont_load`ed",
               not_in_loaded,
               indent=" - ") + "\n" +  # Special indent for tests.
            pp("Params in checkpoint but not in model (code) and not "
               "`dont_load`ed",
               not_in_inited,
               indent=" + "))  # Special indent for tests.

    return u.recover_tree(merged.keys(), merged.values())


class AddPositionEmbs(nn.Module):
    """Adds positional embeddings to the inputs, supports caching for decode.

    Attributes:
        decode: whether to run in single-position autoregressive mode.
    """
    decode: bool = False

    @nn.compact
    def __call__(self, inputs, posemb):
        """Applies AddPositionEmbs module.

        Adds posemb to the inputs, supports single-position autoregressive mode.

        Args:
            inputs: input data [batch_size, seq_len, emb_dim].
            posemb: positional embeddings.

        Returns:
            output: inputs modulated by pos-embeddings
                [batch_size, seq_len, emb_dim].
        """
        assert inputs.ndim == 3, f"Unexpected inputs shape: {inputs.shape}"
        _, seq_len, emb_dim = inputs.shape
        pe = posemb[:, :seq_len, :]

        if self.decode:
            is_initialized = self.has_variable("cache", "cache_index")
            # We use a cache position index for tracking decoding position.
            cache_index = self.variable("cache", "cache_index",
                                        lambda: jnp.array(0, dtype=jnp.uint32))
            if is_initialized:
                i = cache_index.value
                cache_index.value = i + 1
                # Returns posemb[0, i, :], the positional embedding for the
                # current decoding position.
                pe = jax.lax.dynamic_slice(posemb,
                                           start_indices=jnp.array((0, i, 0)),
                                           slice_sizes=(1, 1, emb_dim))
        return inputs + pe


#=============================================================================#
#                           big_vision/models/vit.py                          #
#=============================================================================#

"""A refactored and simplified ViT.

However, the names of modules are made to match the old ones for easy loading.
"""

from typing import Optional, Sequence, Union

from absl import logging
from big_vision import utils
from big_vision.models import common
import flax
import flax.linen as nn
import flax.training.checkpoints
import jax.numpy as jnp
import numpy as np
import scipy.ndimage


def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1. / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate(
        [jnp.sin(x), jnp.cos(x),
         jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(name,
                          nn.initializers.normal(stddev=1 / np.sqrt(width)),
                          (1, np.prod(seqshape), width), dtype)
    elif typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    else:
        raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=True):
        """Applies Transformer MlpBlock module."""
        inits = dict(
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )

        n, l, d = x.shape  # pylint: disable=unused-variable
        x = nn.Dense(self.mlp_dim or 4 * d, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        x = nn.Dense(d, **inits)(x)
        return x


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=True):
        out = {}
        y = nn.LayerNorm()(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
        )(y, y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm()(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
        )(y, deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        return x, out


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""
    depth: int
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=True):
        out = {}

        # Input Encoder
        for lyr in range(self.depth):
            block = Encoder1DBlock(name=f"encoderblock_{lyr}",
                                   mlp_dim=self.mlp_dim,
                                   num_heads=self.num_heads,
                                   dropout=self.dropout)
            x, out[f"block{lyr:02d}"] = block(x, deterministic)
        out["pre_ln"] = x  # Alias for last block, but without the number in it.

        return nn.LayerNorm(name="encoder_norm")(x), out


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12

    @nn.compact
    def __call__(self, x):
        # TODO
        n, l, d = x.shape  # pylint: disable=unused-variable
        probe = self.param("probe", nn.initializers.xavier_uniform(),
                           (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform())(probe, x)

        # TODO: dropout on head?
        y = nn.LayerNorm()(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
        return x[:, 0]


class _Model(nn.Module):
    """ViT model."""

    num_classes: Optional[int] = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    posemb: str = "learn"  # Can also be "sincos2d"
    rep_size: Union[int, bool] = False
    dropout: float = 0.0
    pool_type: str = "gap"  # Can also be "map" or "tok"
    head_zeroinit: bool = True

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        # Patch extraction
        x = out["stem"] = nn.Conv(self.width,
                                  self.patch_size,
                                  strides=self.patch_size,
                                  padding="VALID",
                                  name="embedding")(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # Add posemb before adding extra token.
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c,
                                                "pos_embedding", x.dtype)

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, l, c = x.shape  # pylint: disable=unused-variable
        x = nn.Dropout(rate=self.dropout)(x, not train)

        x, out["encoder"] = Encoder(depth=self.depth,
                                    mlp_dim=self.mlp_dim,
                                    num_heads=self.num_heads,
                                    dropout=self.dropout,
                                    name="Transformer")(
                                        x, deterministic=not train)
        encoded = out["encoded"] = x

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(num_heads=self.num_heads,
                                            mlp_dim=self.mlp_dim)(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, [n, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, name="pre_logits")
            # NOTE: In the past we did not include tanh in pre_logits.
            # For few-shot, it should not matter much, as it whitens anyways.
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {
                "kernel_init": nn.initializers.zeros
            } if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


def Model(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name
    """Factory function, because linen really don't like what I'm doing!"""
    return _Model(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "H": 1280,
            "g": 1408,
            "G": 1664,
            "e": 1792
        }[v],
        "depth": {
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "H": 32,
            "g": 40,
            "G": 48,
            "e": 56
        }[v],
        "mlp_dim": {
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "H": 5120,
            "g": 6144,
            "G": 8192,
            "e": 15360
        }[v],
        "num_heads": {
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "H": 16,
            "g": 16,
            "G": 16,
            "e": 16
        }[v],
        # pylint:enable=line-too-long
        **patch
    }


def resample_posemb(old, new):
    """This function implements "high-res finetuning" for transformer models."""
    # Rescale the grid of position embeddings. Param shape is (1,N,1024)
    if old.shape == new.shape:
        return old

    logging.info("ViT: resize %s to %s", old.shape, new.shape)
    gs_old = int(np.sqrt(old.shape[1]))
    gs_new = int(np.sqrt(new.shape[1]))
    logging.info("ViT: grid-size from %s to %s", gs_old, gs_new)
    grid = old.reshape(gs_old, gs_old, -1)

    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    grid = scipy.ndimage.zoom(grid, zoom, order=1)
    grid = grid.reshape(1, gs_new * gs_new, -1)
    return jnp.array(grid)


def fix_old_checkpoints(params):
    """Fix small bwd incompat that can't be resolved with names in model def."""

    params = flax.core.unfreeze(
        flax.training.checkpoints.convert_pre_linen(params))

    # Original ViT paper variant had posemb in a module:
    if "posembed_input" in params["Transformer"]:
        logging.info("ViT: Loading and fixing VERY old posemb")
        posemb = params["Transformer"].pop("posembed_input")
        params["pos_embedding"] = posemb["pos_embedding"]

    # Widely used version before 2022 had posemb in Encoder:
    if "pos_embedding" in params["Transformer"]:
        logging.info("ViT: Loading and fixing old posemb")
        params["pos_embedding"] = params["Transformer"].pop("pos_embedding")

    # Old vit.py used to first concat [cls] token, then add posemb.
    # This means a B/32@224px would have 7x7+1 posembs. This is useless and clumsy
    # so we changed to add posemb then concat [cls]. We can recover the old
    # checkpoint by manually summing [cls] token and its posemb entry.
    if "pos_embedding" in params:
        pe = params["pos_embedding"]
        if int(np.sqrt(pe.shape[1]))**2 + 1 == int(pe.shape[1]):
            logging.info("ViT: Loading and fixing combined cls+posemb")
            pe_cls, params["pos_embedding"] = pe[:, :1], pe[:, 1:]
            if "cls" in params:
                params["cls"] += pe_cls

    # MAP-head variants during ViT-G development had it inlined:
    if "probe" in params:
        params["MAPHead_0"] = {
            k: params.pop(k)
            for k in [
                "probe", "MlpBlock_0", "MultiHeadDotProductAttention_0",
                "LayerNorm_0"
            ]
        }

    return params


def load(init_params, init_file, model_cfg, dont_load=()):  # pylint: disable=invalid-name because we had to CamelCase above.
    """Load init from checkpoint, both old model and this one. +Hi-res posemb."""
    del model_cfg

    init_file = VANITY_NAMES.get(init_file, init_file)
    restored_params = utils.load_params(None, init_file)

    restored_params = fix_old_checkpoints(restored_params)

    # possibly use the random init for some of the params (such as, the head).
    restored_params = common.merge_params(restored_params, init_params,
                                          dont_load)

    # resample posemb if needed.
    if init_params and "pos_embedding" in init_params:
        restored_params["pos_embedding"] = resample_posemb(
            old=restored_params["pos_embedding"],
            new=init_params["pos_embedding"])

    return restored_params


# Shortcut names for some canonical paper checkpoints:
VANITY_NAMES = {
    # pylint: disable=line-too-long
    # pylint: disable=line-too-long
    # Recommended models from https://arxiv.org/abs/2106.10270
    # Many more models at https://github.com/google-research/vision_transformer
    "howto-i21k-Ti/16":
    "gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-S/32":
    "gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-S/16":
    "gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-B/32":
    "gs://vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/16":
    "gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/8":
    "gs://vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-L/16":
    "gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz",

    # Better plain vit-s16 baselines from https://arxiv.org/abs/2205.01580
    "i1k-s16-90ep": "gs://big_vision/vit_s16_i1k_90ep.npz",
    "i1k-s16-150ep": "gs://big_vision/vit_s16_i1k_150ep.npz",
    "i1k-s16-300ep": "gs://big_vision/vit_s16_i1k_300ep.npz",

    # DeiT-3 checkpoints from https://github.com/facebookresearch/deit/blob/main/README_revenge.md
    # First layer converted to take inputs in [-1,1]
    "deit3_S_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_1k.npz",
    "deit3_S_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_21k.npz",
    "deit3_S_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_1k.npz",
    "deit3_S_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_21k.npz",
    "deit3_B_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_1k.npz",
    "deit3_B_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_21k.npz",
    "deit3_B_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_1k.npz",
    "deit3_B_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_21k.npz",
    "deit3_L_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_1k.npz",
    "deit3_L_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_21k.npz",
    "deit3_L_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_1k.npz",
    "deit3_L_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_21k.npz",
    # pylint: disable=line-too-long
    # pylint: enable=line-too-long
}


#=============================================================================#
#                                                                             #
#                       ██    ██ ██    ██ ██ ███    ███                       #
#                       ██    ██ ██    ██ ██ ████  ████                       #
#                       ██    ██ ██    ██ ██ ██ ████ ██                       #
#                       ██    ██  ██  ██  ██ ██  ██  ██                       #
#                        ██████    ████   ██ ██      ██                       #
#                                                                             #
#=============================================================================#

#=============================================================================#
#                      big_vision/models/proj/uvim/vit.py                     #
#=============================================================================#
#==== STAGE I

"""VQ-VAE autoencoder with ViT backbone."""

import functools
from typing import Mapping, Optional, Sequence, Union

from big_vision import utils
from big_vision.models import common
from big_vision.models import vit

import einops
import flax.linen as nn
import flax.training.checkpoints
import jax
import jax.numpy as jnp
import numpy as np

partial = functools.partial

# Multiplicative perturbation applied to codewords when doing the split.
# Note, the multiplicative pertubation is not perfectly symmetric and rep.
# applications can shrink the embedding. However, in practice it does not matter
# for the value we use.
PERTURB = 0.001


# The function below takes a vector `x` and a dictioniary of vectors `e` as an
# input. It then returns a "quantized" version of x (namely the closest to `x`
# vector from `e`) and its index in `e` as well.
# On top of this, it has two extra features:
#   1. Double `vmap` vectorizes this function to operate on many `x` vectors.
#      More concretely, we add two extra dimensions (batch and space) to `x`.
#      Also note we compute euclidian distance in a decomposed way, because it
#      makes it more efficient for vmapping.
#   2. `quantize` is a "discrete" operation, so it does not have a gradient for
#      `x`. So we implement a so-called "straight-through" gradient estimator
#      using `stop_gradient` magic. It does not affect forward pass, but changes
#      the gradient.
@partial(jax.vmap, in_axes=(0, None), out_axes=(0, 0))
@partial(jax.vmap, in_axes=(0, None), out_axes=(0, 0))
def quantize(x, e):
    dist = jnp.sum(x * x)[None] - 2 * x.dot(e.T) + jnp.sum(e * e, axis=1)
    idx = jnp.argmin(dist)
    x_q = jax.lax.stop_gradient(e[idx] -
                                x) + x  # just `e[idx]` for the fwd pass.
    return x_q, idx


def split_the_most_frequent_embedding(state):
    """Splits most frequent embedding into two and eliminates least frequent.

    Args:
        state: a dict. that contains current jax rng, embeddings and their counts.

    Returns:
        New dict. with the updated jax rng, embeddings and counts.
    """
    rng, e, c = state["rng"], state["dictionary"], state["counts"]
    rng, rng_local = jax.random.split(rng)

    i_max = jnp.argmax(c)
    i_min = jnp.argmin(c)

    e = e.at[i_min].set(e[i_max] * jax.random.uniform(
        rng_local, (e.shape[1], ), jnp.float32, 1.0 - PERTURB, 1.0 + PERTURB))

    c = c.at[i_min].set(c[i_max] / 2.0)
    c = c.at[i_max].set(c[i_max] / 2.0)

    e = e.at[i_min].set(e[i_min] / 2.0)
    e = e.at[i_max].set(e[i_max] / 2.0)

    return {"rng": rng, "dictionary": e, "counts": c}


class Model(nn.Module):
    """ViT model."""

    inputs: Mapping[str, Sequence[int]]
    outputs: Mapping[str, Sequence[int]]
    input_size: Sequence[int] = (256, 256)
    patch_size: Sequence[int] = (8, 8)
    code_len: int = 256
    width: int = 768
    enc_depth: int = 6
    dec_depth: int = 6
    mlp_dim: Optional[int] = None
    num_heads: int = 12
    posemb: str = "learn"  # Can also be "sincos2d"
    rep_size: Union[int, bool] = False
    dropout: float = 0.0
    reinit: Optional[Sequence[str]] = None
    head_zeroinit: bool = True
    dict_size: int = 512  # Number of words in dict.
    codeword_dim: Optional[int] = None
    dict_momentum: float = 0.995  # Exp. moving average coeff. for dict. learning.
    quantize: bool = True
    # Useful to set to None when running without pmap, e.g. testing.
    statistics_axis_name: str = "batch"
    # Threshold for the discounted count after which the codeword will be
    # considered unused. For the `dict_momentum` param of 0.995 the codeword
    # should not be present in ~500 batches in a row.
    min_count: float = 0.1  # ~= 0.995 ** 500
    with_encoder_ctx: bool = False
    with_decoder_ctx: bool = False
    code_dropout: str = "none"
    bottleneck_resize: bool = False
    zero_decoder_seq: bool = False

    def setup(self):

        self.grid_size = np.array(self.input_size) // np.array(self.patch_size)

        self.embeddings = {
            k:
            nn.DenseGeneral(features=(self.width, ),
                            axis=range(-len(shape), 0),
                            name=f"embedding_{k}")
            for k, shape in self.inputs.items()
        }

        kw = {
            "kernel_init": nn.initializers.zeros
        } if self.head_zeroinit else {}
        self.heads = {
            k: nn.DenseGeneral(features=shape, name=f"head_{k}", **kw)
            for k, shape in self.outputs.items()
        }

        if self.with_encoder_ctx:
            self.stem_conv_ctx_enc = nn.Conv(self.width,
                                             self.patch_size,
                                             strides=self.patch_size,
                                             padding="VALID",
                                             name="ctx_enc_embedding")

        if self.with_decoder_ctx:
            self.stem_conv_ctx_dec = nn.Conv(self.width,
                                             self.patch_size,
                                             strides=self.patch_size,
                                             padding="VALID",
                                             name="ctx_dec_embedding")

        self.pos_embedding_encoder = vit.get_posemb(self, self.posemb,
                                                    self.grid_size, self.width,
                                                    "pos_embedding_encoder")
        self.encoder = vit.Encoder(depth=self.enc_depth,
                                   mlp_dim=self.mlp_dim,
                                   num_heads=self.num_heads,
                                   dropout=self.dropout,
                                   name="encoder")

        if not self.bottleneck_resize:
            self.bottleneck_downsample = self.param(
                "bottleneck_downsample", nn.initializers.xavier_uniform(),
                (np.prod(self.grid_size), self.code_len))

        norm_init = nn.initializers.normal(stddev=1.0 /
                                           np.sqrt(self.dict_size))
        self.dictionary = self.variable(
            "state", "dictionary",
            lambda shape: norm_init(self.make_rng("state"), shape),
            (self.dict_size, self.codeword_dim or self.width))
        self.counts = self.variable("state", "counts", jnp.ones,
                                    (self.dict_size, ))

        if not self.bottleneck_resize:
            self.bottleneck_upsample = self.param(
                "bottleneck_upsample", nn.initializers.xavier_uniform(),
                (self.code_len, np.prod(self.grid_size)))

        self.pos_embedding_decoder = vit.get_posemb(self, self.posemb,
                                                    self.grid_size, self.width,
                                                    "pos_embedding_decoder")
        self.decoder = vit.Encoder(depth=self.dec_depth,
                                   mlp_dim=self.mlp_dim,
                                   num_heads=self.num_heads,
                                   dropout=self.dropout,
                                   name="decoder")

        self.encoder_head = nn.Dense(self.codeword_dim or self.width)
        self.decoder_stem = nn.Dense(self.width)

    def get_codewords(self):
        e = self.dictionary.value / self.counts.value[:, None]
        e = e / jnp.linalg.norm(e, axis=-1, keepdims=True)
        return e

    def encode(self, x, *, ctx=None, train=False, update_dict=True):
        out = {}

        out["stem"] = {}
        for key, embed in self.embeddings.items():
            out["stem"][key] = embed(x[key])
        x = sum(out["stem"].values())

        if self.with_encoder_ctx:
            ctx_tokens = self.stem_conv_ctx_enc(ctx)
            ctx_tokens = einops.rearrange(ctx_tokens, "b h w c -> b (h w) c")
            x = x + ctx_tokens

        x, _ = self.encoder(x + self.pos_embedding_encoder,
                            deterministic=not train)

        if self.bottleneck_resize:
            x = einops.rearrange(x,
                                 "b (h w) c -> b h w c",
                                 h=self.grid_size[0],
                                 w=self.grid_size[1])
            l = int(np.round(self.code_len**0.5))
            x = jax.image.resize(x, (x.shape[0], l, l, x.shape[3]),
                                 method="linear")
            x = einops.rearrange(x, "b h w c -> b (h w) c")
        else:
            x = jnp.einsum("btc,tn->bnc", x, self.bottleneck_downsample)

        x = self.encoder_head(x)

        x = jax.nn.standardize(x, axis=-1)
        x_pre_q = out["bottleneck"] = x
        e = self.get_codewords()
        x, idx = quantize(x, e)
        out["bottleneck_q"] = x
        out["code"] = idx

        # Implements explicit dictionary learning algo outlined in the VQ-VAE paper.
        # We slightly deviate from the papers formulation, as we find it confusing,
        # especially in the multi-host scenario. What is implemented below can be
        # seen as computing discounted counts and sums of all embeddings.
        if train:
            # Compute counts and sum(x) of code in the global batch.
            counts = jnp.zeros(self.dict_size, dtype=jnp.int32)
            counts = counts.at[idx].add(1)

            # Below we introduce redundant stop_gradient, because jax' dead code
            # elimination for our program's gradient fails to infer that the code
            # below does not require gradient computation.
            # Relevant github issue: https://github.com/google/jax/issues/9042.
            # TODO: remove stop_gradient when the bug is fixed.
            x_sum = jnp.zeros_like(self.dictionary.value)
            x_sum = x_sum.at[idx].add(jax.lax.stop_gradient(x_pre_q))

            if self.statistics_axis_name:
                counts = jax.lax.psum(counts,
                                      axis_name=self.statistics_axis_name)
                x_sum = jax.lax.psum(x_sum,
                                     axis_name=self.statistics_axis_name)

            out["codebook_max_ratio"] = jnp.max(counts) / jnp.sum(counts)
            out["codebook_zeros_ratio"] = jnp.sum(counts == 0) / len(counts)

            if update_dict:
                self.counts.value = self.counts.value * self.dict_momentum + counts
                self.dictionary.value = (
                    self.dictionary.value * self.dict_momentum + x_sum)

                state = {
                    "dictionary": self.dictionary.value,
                    "counts": self.counts.value,
                    "rng": self.make_rng("vqvae")
                }
                new_state = jax.lax.while_loop(
                    lambda state: jnp.any(state["counts"] < self.min_count),
                    split_the_most_frequent_embedding, state)
                self.counts.value = new_state["counts"]
                self.dictionary.value = new_state["dictionary"]

        if not self.quantize:
            x = x_pre_q
            out["bottleneck_q"] = x
        return x, out

    def decode(self, x, ctx=None, discrete_input=False, train=False):
        out = {}

        if discrete_input:
            e = self.get_codewords()
            x = e[x]

        if self.zero_decoder_seq:
            x = jnp.zeros_like(x)

        if train and self.code_dropout != "none":
            importance = jnp.linspace(1.0, 0.0, self.code_len + 2)[1:-1]
            thr = jax.random.uniform(self.make_rng("dropout"), x.shape[:1])
            mask = importance[None, :] > thr[:, None]
            if self.code_dropout == "random":
                mask = jax.random.permutation(self.make_rng("dropout"),
                                              mask,
                                              axis=-1,
                                              independent=True)
            x = x * mask[:, :, None]

        x = self.decoder_stem(x)

        if self.bottleneck_resize:
            l = int(np.round(self.code_len**0.5))
            x = einops.rearrange(x, "b (h w) c -> b h w c", h=l, w=l)
            x = jax.image.resize(
                x,
                (x.shape[0], self.grid_size[0], self.grid_size[1], x.shape[3]),
                method="linear")
            x = einops.rearrange(x, "b h w c -> b (h w) c")
        else:
            x = jnp.einsum("bnc,nt->btc", x, self.bottleneck_upsample)

        if self.with_decoder_ctx:
            ctx_tokens = self.stem_conv_ctx_dec(ctx)
            ctx_tokens = einops.rearrange(ctx_tokens, "b h w c -> b (h w) c")
            x = x + ctx_tokens

        x, _ = self.decoder(x + self.pos_embedding_decoder)

        out["logits"] = {}
        for key, head in self.heads.items():
            out["logits"][key] = head(x)

        return out["logits"], out

    def __call__(self, x, *, ctx=None, train=False, update_dict=True):
        x, out_enc = self.encode(x,
                                 ctx=ctx,
                                 train=train,
                                 update_dict=update_dict)
        x, out_dec = self.decode(x, ctx=ctx, train=train)
        return x, {**out_enc, **out_dec}


def load(init_params, init_file, model_params=None, dont_load=()):
    """Loads params from init checkpoint and merges into init_params."""
    del model_params
    ckpt = flax.core.unfreeze(utils.load_checkpoint(None, init_file))
    params = {"params": ckpt["params"], "state": ckpt["state"]}
    params = flax.training.checkpoints.convert_pre_linen(params)
    # Fix old-style param name.
    if "Encoder" in params["params"]:
        p = params["params"]
        p["encoder"] = p.pop("Encoder")
        p["decoder"] = p.pop("Decoder")
        params["params"] = p
    if init_params is not None:
        params = common.merge_params(params, init_params, dont_load)
    return params["params"], params["state"]


#=============================================================================#
#                      big_vision/models/proj/uvim/vtt.py                     #
#=============================================================================#
#==== STAGE II

"""Simple vision-text transformer with encoder-decoder architecture.

Used abbreviations for dimension annotations:
  B: batch size.
  H: image height.
  W: image width.
  P: number of patches (PH/PW: number of patches in height/width dimensions).
  E: embedding size.
  L: sequence length of text tokens.
  V: vocab size.
"""
from typing import Sequence
from big_vision import utils
from big_vision.models import common
from big_vision.models import vit
import einops
import flax
import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import numpy as np


def shift_right(x, axis=1):
    """Shift to the right on given axis with padding value 0."""
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    padded = jnp.pad(x, pad_widths, constant_values=0)
    return padded[:, :-1]


class EncoderDecoderBlock(nn.Module):
    """Transformer encoder-decoder layer."""
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.
    decode: bool = False

    @nn.compact
    def __call__(self,
                 targets,
                 encoded,
                 decoder_mask=None,
                 deterministic=True):
        """Applies EncoderDecoder1DBlock module.

    Args:
      targets: target text embeddings [B, L, E].
      encoded: encoded image patches from encoder [B, P, E].
      decoder_mask: decoder self-attention mask.
      deterministic: bool, deterministic or not (to apply dropout).

    Returns:
      output after transformer encoder-decoder block [B, L, E].
    """
        # Decoder block.
        x = nn.LayerNorm(name="LayerNorm1")(targets)
        x = nn.SelfAttention(num_heads=self.num_heads,
                             use_bias=False,
                             broadcast_dropout=False,
                             dropout_rate=self.dropout_rate,
                             decode=self.decode,
                             name="SelfAttn")(x,
                                              decoder_mask,
                                              deterministic=deterministic)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + targets

        # Encoder-Decoder block.
        y = nn.LayerNorm(name="LayerNorm2")(x)
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads,
                                            use_bias=False,
                                            broadcast_dropout=False,
                                            dropout_rate=self.dropout_rate,
                                            name="CrossAttn")(
                                                y,
                                                encoded,
                                                deterministic=deterministic)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = y + x

        # MLP block.
        z = nn.LayerNorm(name="LayerNorm3")(y)
        z = vit.MlpBlock(mlp_dim=self.mlp_dim,
                         dropout=self.dropout_rate,
                         name="MLP")(z, deterministic=deterministic)

        return y + z


class Decoder(nn.Module):
    """Transformer Model Decoder for sequence to sequence translation."""
    emb_dim: int
    mlp_dim: int
    num_heads: int
    num_layers: int
    dropout_rate: float = 0.
    output_vocab_size: int = 32000
    zero_decoder_seq: bool = False

    @nn.compact
    def __call__(self,
                 encoded,
                 targets,
                 pos_emb,
                 decoder_mask=None,
                 decode=False,
                 deterministic=True,
                 max_decode_length=None):
        """Applies Transformer model on the inputs.

        Args:
            encoded: encoded image patches from encoder [B, P, E].
            targets: target text tokens [B, L].
            pos_emb: positional embeddings.
            decoder_mask: decoder self-attention mask.
            decode: bool, whether to perform fast autoregressive
                decoding with cache.
            deterministic: bool, deterministic or not (to apply dropout).
            max_decode_length: optional max length for positional embeddings.

        Returns:
            output of a transformer decoder [B, L, V].
        """
        y = targets.astype("int32")
        if not decode:
            y = shift_right(y)
        y = nn.Embed(self.output_vocab_size,
                     self.emb_dim,
                     name="EmbedTargets",
                     embedding_init=nn.initializers.normal(stddev=1.0))(y)
        if self.zero_decoder_seq:
            y = jnp.zeros_like(y)
        y = common.AddPositionEmbs(decode=decode,
                                   name="PosEmbedTargets")(y, pos_emb)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)

        for lyr in range(self.num_layers):
            y = EncoderDecoderBlock(num_heads=self.num_heads,
                                    mlp_dim=self.mlp_dim,
                                    dropout_rate=self.dropout_rate,
                                    decode=decode,
                                    name=f"EncDecBlock{lyr}")(
                                        y,
                                        encoded,
                                        decoder_mask=decoder_mask,
                                        deterministic=deterministic)
        y = nn.LayerNorm(name="LayerNorm")(y)
        logits = nn.Dense(self.output_vocab_size,
                          kernel_init=nn.initializers.zeros,
                          name="LogitsDense")(y)
        return logits


class Model(nn.Module):
    """Transformer Model for sequence to sequence translation."""
    patches: ml_collections.ConfigDict
    # Encoder/decoder shared params:
    num_heads: int = 8
    num_layers: int = 6
    mlp_dim: int = 2048
    dropout_rate: float = 0.
    # Decoder params:
    emb_dim: int = 512
    vocab_size: int = 32000
    seq_len: int = 256
    # Encoder params:
    input_size: Sequence[int] = (256, 256)
    posemb_type: str = "sincos2d"  # Can also be "learn"
    zero_decoder_seq: bool = False

    def setup(self):
        grid_size = np.array(self.input_size) // np.array(self.patches.size)
        self.pos_emb_for_encoder = vit.get_posemb(self, self.posemb_type,
                                                  grid_size, self.emb_dim,
                                                  "pos_embedding_encoder")
        self.pos_emb_for_decoder = vit.get_posemb(self, self.posemb_type,
                                                  (1, self.seq_len),
                                                  self.emb_dim,
                                                  "pos_embedding_decoder")

        self.encoder = vit.Encoder(depth=self.num_layers,
                                   mlp_dim=self.mlp_dim,
                                   num_heads=self.num_heads,
                                   dropout=self.dropout_rate)
        self.decoder = Decoder(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            emb_dim=self.emb_dim,
            output_vocab_size=self.vocab_size,
            zero_decoder_seq=self.zero_decoder_seq,
        )
        self.conv = nn.Conv(self.emb_dim,
                            self.patches.size,
                            padding="VALID",
                            strides=self.patches.size,
                            name="EmbedPatches")

    def encode(self, image, train=False):
        """Encodes input image or embeddings."""
        emb = self.conv(image)
        patch_embeddings = einops.rearrange(emb, "B PH PW E -> B (PH PW) E")
        encoded, _ = self.encoder(patch_embeddings + self.pos_emb_for_encoder,
                                  deterministic=not train)
        return encoded

    def decode(self,
               encoded,
               targets,
               decode=False,
               train=False,
               max_decode_length=None):
        """Applies Transformer decoder-branch on encoded-input and target.

        Args:
            encoded: encoded image patches from encoder [B, P, E].
            targets: target text tokens [B, L].
            decode: whether to prepare and use an autoregressive cache.
            train: whether it is training.
            max_decode_length: optional max length for positional embeddings.

        Returns:
            logits array from transformer decoder [B, L, V].
    """
        decoder_mask = None if decode else nn.make_causal_mask(targets)
        logits = self.decoder(encoded,
                              targets,
                              pos_emb=self.pos_emb_for_decoder,
                              decoder_mask=decoder_mask,
                              decode=decode,
                              deterministic=not train,
                              max_decode_length=max_decode_length)
        return logits

    def __call__(self, image, text, *, decode=False, train=False):
        """Applies Transformer model on the inputs.

        Args:
            image: batch of images [B, H, W, 3].
            text: batch of tokenized texts [B, L].
            decode: whether to prepare and use an autoregressive cache.
            train: whether it is training.

        Returns:
            logits array from full transformer [B, L, V].
    """
        encoded = self.encode(image, train=train)
        return self.decode(encoded, text, decode=decode, train=train)


def load(init_params,
         init_files,
         model_params=None,
         dont_load=("head/kernel", "head/bias", "cls")):
    """Loads params from init checkpoint and merges into init_params."""
    del model_params
    if isinstance(init_files, str):
        # A shortcut for a single file checkpoint of a vtt model.
        ckpt_params = utils.load_params(None, init_files)
        ckpt_params = flax.training.checkpoints.convert_pre_linen(ckpt_params)
        if init_params is not None:
            ckpt_params = common.merge_params(ckpt_params, init_params,
                                              dont_load)
    else:
        init_files = {
            **init_files
        }  # Shallow copy because we'll pop stuff off.

        enc_init = init_files.pop("encoder", None)
        if enc_init:
            ckpt_params = init_params.copy()
            vit_params = {
                "pos_embedding": ckpt_params["pos_embedding_encoder"],
                "Transformer": ckpt_params["encoder"],
                "embedding": ckpt_params["EmbedPatches"],
            }
            encoder_params = vit.load(vit_params,
                                      enc_init,
                                      model_cfg={},
                                      dont_load=dont_load)
            ckpt_params["encoder"] = encoder_params["Transformer"]
            ckpt_params["pos_embedding_encoder"] = encoder_params[
                "pos_embedding"]
            ckpt_params["EmbedPatches"] = encoder_params["embedding"]
        else:
            raise ValueError(
                "Only encoder init is supported: {}.".format(init_files))

    return ckpt_params


#=============================================================================#
#                                                                             #
#                    ██ ███    ██ ███████ ███████ ██████                      #
#                    ██ ████   ██ ██      ██      ██   ██                     #
#                    ██ ██ ██  ██ █████   █████   ██████                      #
#                    ██ ██  ██ ██ ██      ██      ██   ██                     #
#                    ██ ██   ████ ██      ███████ ██   ██                     #
#                                                                             #
#=============================================================================#
## big_vision/models/proj/uvim/decode.py

"""Inference."""
import functools

from typing import Any, Callable, Optional, Tuple

import flax
from flax import linen as nn
import jax
from jax import lax
from jax import numpy as jnp

import numpy as np

EOS_ID = 1
NEG_INF = np.array(-1.0e7)  # Effective negative infinity.

GenerateFn = Callable[..., Tuple[jnp.ndarray, jnp.ndarray,
                                 Optional[jnp.ndarray]]]


def temperature_sampling(*args, temperature=1.0, top_k=0, top_p=0.0, **kwargs):
    """Convenience wrapper for temperature sampling."""
    return generate(*args,
                    generate_fn=_temperature_sampling,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    **kwargs)


def topk_sampling(*args, temperature=1.0, top_k=20, **kwargs):
    """Convenience wrapper for top-k sampling."""
    return generate(*args,
                    generate_fn=_temperature_sampling,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=0.0,
                    **kwargs)


def nucleus_sampling(*args, temperature=1.0, top_p=0.2, **kwargs):
    """Convenience wrapper for nucleus sampling."""
    return generate(*args,
                    generate_fn=_temperature_sampling,
                    temperature=temperature,
                    top_k=0,
                    top_p=top_p,
                    **kwargs)


def argmax_sampling(*args, **kwargs):
    """Convenience wrapper for argmax sampling."""
    return generate(*args,
                    generate_fn=_temperature_sampling,
                    temperature=1e-7,
                    top_k=0,
                    top_p=0.0,
                    **kwargs)


def generate(params,
             inputs,
             prompts,
             seed,
             *,
             model: nn.Module,
             generate_fn: GenerateFn,
             num_samples: int = 1,
             prefill: bool = False,
             eos_token: int = EOS_ID,
             **generate_fn_kwargs):
    """Generate sequence with fast decoding beam search on a batch.

    Model must support:
        encode(inputs) -> encoded, or encode(*inputs) -> encoded.
        decode(encoded, prompts, decode=True/False, max_decode_length) -> logits

    Args:
        params: model parameters.
        inputs: either a single `jnp.ndarray` of e.g. images, or
            a tuple of inputs which are passed via `model.encode(*inputs)`.
        prompts: [batch_size, max_decode_len] forced tokens for generation.
            prompts need to finish with 0 token, they should not contain the end
            markers. If no prompting is required, pass an all zeros tensor.
        seed: PRNG key for random sampling.
        model: object with methods encode and decode.
        generate_fn: search or sampling function to generate sequences.
        num_samples: number of samples to generate per item.
        prefill: whether to prefill cache.
        eos_token: if of end-of-sentence token for target vocabulary.
        **generate_fn_kwargs: generate fn specific kwargs.

    Returns:
        Top-scoring sequences (worst scores first).
            [batch_size, num_samples, max_decode_len]
        Scores of the generated sequences (worst scores first). The
            returned scores are modified log probabilities. May be absent.
            [batch_size, max_decode_len]
        Log probs for the generated tokens. May be absent.
            [batch_size, num_samples, max_decode_len]
    """
    _, max_decode_len = prompts.shape
    decode_kwargs = {"max_decode_length": max_decode_len}

    def encode(model, inputs):
        if not isinstance(inputs, tuple):
            inputs = (inputs, )
        return model.encode(*inputs)

    encoded_inputs = nn.apply(encode, model)(params, inputs)
    if isinstance(encoded_inputs, tuple):
        encoded_inputs, enc_pos_emb = encoded_inputs
        decode_kwargs["enc_pos_emb"] = enc_pos_emb

    def init_cache(model):
        encoded = jnp.zeros_like(encoded_inputs)
        targets = jnp.zeros_like(prompts)
        return model.decode(encoded, targets, decode=True, **decode_kwargs)

    cache = nn.apply(init_cache, model, mutable=True)(params)[1]["cache"]

    def prefill_cache(model, encoded, targets):
        return model.decode(encoded, targets, prefill=True, **decode_kwargs)

    if prefill:
        cache = nn.apply(prefill_cache, model,
                         mutable=True)({
                             "params": params["params"],
                             "cache": cache
                         }, encoded_inputs, prompts)[1]["cache"]

    def tokens_to_logits(tokens, cache):

        def decode_step(model, tokens):
            encoded = expand_samples_dim_and_flatten(encoded_inputs,
                                                     num_samples)
            return model.decode(encoded, tokens, decode=True, **decode_kwargs)

        logits, aux = nn.apply(decode_step, model,
                               mutable=True)({
                                   "params": params["params"],
                                   "cache": cache
                               }, tokens)
        return logits.squeeze(axis=1), aux["cache"]

    beam_seqs, scores, logprobs = generate_fn(prompts,
                                              cache,
                                              tokens_to_logits,
                                              num_samples=num_samples,
                                              eos_token=eos_token,
                                              max_decode_len=max_decode_len,
                                              seed=seed,
                                              **generate_fn_kwargs)
    return beam_seqs, scores, logprobs


def expand_samples_dim(x, num_samples):
    """Creates new dimension in non-scalar array and tiles into it."""
    if x.ndim == 0:  # ignore scalars (e.g. cache index)
        return x
    x = jnp.expand_dims(x, axis=1)
    tile_dims = [1] * x.ndim
    tile_dims[1] = num_samples
    return jnp.tile(x, tile_dims)


def flatten_samples_dim(x):
    """Flattens samples dim into batch dim."""
    if x.ndim == 0:  # ignore scalars (e.g. cache index)
        return x
    return x.reshape((x.shape[0] * x.shape[1], ) + x.shape[2:])


def unflatten_samples_dim(x, batch_size, num_samples):
    """Unflattens first dim into batch and samples dims."""
    if x.ndim == 0:  # ignore scalars (e.g. cache index)
        return x
    assert batch_size * num_samples == x.shape[0]
    return x.reshape((batch_size, num_samples) + x.shape[1:])


def expand_samples_dim_and_flatten(x, num_samples):
    """Expands the each batch item by num_samples in batch dimension."""
    return flatten_samples_dim(expand_samples_dim(x, num_samples))


def cache_map(fn, cache):
    """Maps function over caches, even multiple caches in various layers."""
    frozen = isinstance(cache, flax.core.FrozenDict)
    if frozen:
        cache = flax.core.unfreeze(cache)
    flat_cache = flax.traverse_util.flatten_dict(cache)
    # Exclude cached relative position bias from beam expansion, etc.
    keyvals = {k: v for k, v in flat_cache.items() if k[-1] != "cached_bias"}
    keyvals = jax.tree_map(fn, keyvals)
    flat_cache.update(keyvals)
    new_cache = flax.traverse_util.unflatten_dict(flat_cache)
    if frozen:
        new_cache = flax.core.freeze(new_cache)
    return new_cache


@flax.struct.dataclass
class LoopState:
    """Internal state of the temperature sampling loop."""
    # Position in the sequence that we are currently looking at.
    cur_index: int
    # Cache for fast auto-regressive decoding.
    cache: Any
    # Flags indicating whether the sequence reached eos [B*N].
    flags_finished: jnp.ndarray
    # Sequences being generated [B*N, L+1]. Note: sequences start with 0 token.
    sequences: jnp.ndarray
    scores: jnp.array  # Total sequence scores per batch element [B*N].
    logprobs: jnp.array  # Logprobs of selected tokens [B*N, L].
    rng: jnp.ndarray  # PRNGKey of the loop state.


def _init_state(prompts, cache, init_rng_key, num_samples):
    batch_size, max_decode_len_plus_one = prompts.shape
    # Add extra samples dim to attention cache pytree elements.
    cache = cache_map(lambda x: expand_samples_dim_and_flatten(x, num_samples),
                      cache)
    return LoopState(
        cur_index=0,
        cache=cache,
        flags_finished=jnp.zeros((batch_size * num_samples), dtype=jnp.bool_),
        sequences=expand_samples_dim_and_flatten(prompts, num_samples),
        scores=jnp.zeros((batch_size * num_samples)),
        logprobs=jnp.zeros(
            (batch_size * num_samples, max_decode_len_plus_one - 1)),
        rng=init_rng_key)


def _should_temperature_sampling_continue(state, max_decode_len):
    """Check if we should continue or not."""

    max_length_not_reached = state.cur_index < max_decode_len - 1
    all_seqs_finished = jnp.all(state.flags_finished)
    return max_length_not_reached & (~all_seqs_finished)


def _temperature_sampling_iteration(state,
                                    tokens_to_logits,
                                    temperature,
                                    eos,
                                    top_k,
                                    top_p,
                                    mask_token_ids=()):
    """Temperature sampling step function."""

    rng_sampling, rng = jax.random.split(state.rng)

    # 1. Use the model to generate a distribution over the vocabulary (for the
    # next token) and sample from it, optionally applying the temperature.
    # --> [B,].
    cur_tokens = state.sequences[:, state.cur_index]
    logits, new_cache = tokens_to_logits(cur_tokens[:, None], state.cache)
    assert logits.ndim == 2, ("tokens_to_logits expected to return a"
                              f"2-dimensional array [B, V], got {logits.ndim}"
                              "dimensions.")
    logprobs = jax.nn.log_softmax(logits)

    # Do not sample special tokens in with ids in mask_token_ids.
    if mask_token_ids:
        probs = jax.nn.softmax(logits)
        for i in mask_token_ids:
            probs = probs.at[:, i].set(0.)
        probs = probs / jnp.sum(probs, -1, keepdims=True)
        logits = jnp.log(probs)

    if top_p:  # Nucleus sampling.
        logits_sorted = jnp.sort(logits, axis=-1)[:, ::-1]
        sorted_cum_probs = jnp.cumsum(jax.nn.softmax(logits_sorted, axis=-1),
                                      axis=-1)
        cutoff_index = jnp.sum(sorted_cum_probs < top_p,
                               axis=-1,
                               keepdims=True)
        cutoff_logit = jnp.take_along_axis(logits_sorted,
                                           cutoff_index,
                                           axis=-1)
        logits = jnp.where(logits < cutoff_logit,
                           jnp.full_like(logits, NEG_INF), logits)
    if top_k:
        topk_logits, topk_indices = jax.lax.top_k(logits, top_k)
        topk_token = jax.random.categorical(rng_sampling,
                                            topk_logits / temperature)
        sampled_tokens = jnp.squeeze(jnp.take_along_axis(topk_indices,
                                                         jnp.expand_dims(
                                                             topk_token, -1),
                                                         axis=-1),
                                     axis=-1)
    else:
        sampled_tokens = jax.random.categorical(rng_sampling,
                                                logits / temperature)

    sampled_logprobs = jnp.squeeze(jnp.take_along_axis(logprobs,
                                                       jnp.expand_dims(
                                                           sampled_tokens,
                                                           axis=1),
                                                       axis=-1),
                                   axis=-1)

    # 2. Use the sampled tokens to update the sequences that did not finish yet,
    # but only if they are out of prompt.
    next_tokens = state.sequences[:, state.cur_index + 1]
    next_logprobs = jnp.squeeze(jnp.take_along_axis(logprobs,
                                                    jnp.expand_dims(
                                                        next_tokens, axis=1),
                                                    axis=-1),
                                axis=-1)
    out_of_prompt = next_tokens == 0
    update_pos = out_of_prompt * (~state.flags_finished)
    next_tokens = sampled_tokens * update_pos + next_tokens * (~update_pos)
    sampled_logprobs = update_pos * sampled_logprobs + ~update_pos * next_logprobs
    sequences = state.sequences.at[:, state.cur_index + 1].set(next_tokens)
    scores = state.scores + sampled_logprobs
    seqs_logprobs = state.logprobs.at[:, state.cur_index].set(sampled_logprobs)

    # 3. Update the finished flags. Only out of prompts seqs can finish.
    flags_finished = out_of_prompt & (state.flags_finished |
                                      (sampled_tokens == eos))
    return LoopState(cur_index=state.cur_index + 1,
                     cache=new_cache,
                     flags_finished=flags_finished,
                     sequences=sequences,
                     scores=scores,
                     logprobs=seqs_logprobs,
                     rng=rng)


def _temperature_sampling(prompts,
                          cache,
                          tokens_to_logits,
                          num_samples=1,
                          eos_token=EOS_ID,
                          max_decode_len=None,
                          seed=0,
                          temperature=1.,
                          top_k=0,
                          top_p=0.0,
                          mask_token_ids=()):
    """Temperature sampling.

    Purely stochastic sampling-based greedy procedure to generate sequences. Every
    next token in the sequence is sampled from the discrete vocab distribution
    produced by the auto-regressive sequence model. Optionally we can adjust the
    distribution by changing the temperature before sampling from it. Generated
    sequences are no longer than max_decode_len.

    Args:
        prompts: optional prompts [B, L]. By default (None), we call free form
            generation without any prompts. Prompt sequences should finish with
            trailing zeros and should not contain eos tokens.
        cache: cache for fast decoding (generation).
        tokens_to_logits: fast autoregressive decoder function taking single token
            slices and cache and returning next-token logits and updated cache.
        num_samples: int: number of samples to generate per batch item. Note, no
            deduplication is performed, and in dependence of parameter settings, same
            sequences could be generated and returned.
        eos_token: end-of-sentence token.
        max_decode_len: maximal length of generated sequences (L).
        seed: PRNGKey for random sampling.
        temperature: positive real-valued sampling temperature. By default we sample
            from the original distribution. As the temperature approaches 0., the
            entire distribution concentrates on the most probable outcome(s).
        top_k: limit sampling to only top-k logits. Zero means no limit.
        top_p: limit sampling to smallest number of top logits with max cumulative
            prob <= top_p. Zero means no limit. Cannot use both top_p and top_k.
        mask_token_ids: if set then tokens with given ids are not sampled.

    Returns:
        sequences: generated sequences [B, num_samples, L].
        scores: not implemented in the naive temperature sampling [B, num_samples].
        logprobs: Log probabilities for the generated tokens [B, num_samples, L].
    """
    if top_k > 0 and top_p > 0.0:
        raise ValueError(f"Cannot use both top_k {top_k} and top_p {top_p}.")
    if max_decode_len is None:
        max_decode_len = prompts.shape[1]
    # We will start generating sequences from 0 token.
    prompts = jnp.pad(prompts, ((0, 0), (1, 0)))
    eos = jnp.array(eos_token)
    if isinstance(seed, int):
        seed = jax.random.PRNGKey(seed)

    # Initialize the state.
    loop_init_state = _init_state(prompts, cache, seed, num_samples)
    should_temperature_sampling_continue_fn = functools.partial(
        _should_temperature_sampling_continue,
        max_decode_len=max_decode_len +
        1)  # Account for prompt padding with 0's.
    temperature_sampling_iteration_fn = functools.partial(
        _temperature_sampling_iteration,
        tokens_to_logits=tokens_to_logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos=eos,
        mask_token_ids=mask_token_ids)

    # Run the temperature sampling and generate the sequences.
    final_state = lax.while_loop(should_temperature_sampling_continue_fn,
                                 temperature_sampling_iteration_fn,
                                 loop_init_state)

    # Return the generated sequences, discarding the 0 token in the beginning.
    return (final_state.sequences[:, 1:].reshape(
        (-1, num_samples, max_decode_len)),
            final_state.scores.reshape((-1, num_samples)),
            final_state.logprobs.reshape((-1, num_samples, max_decode_len)))


#=============================================================================#
#                                                                             #
#                    ████████ ██████   █████  ██ ███    ██                    #
#                       ██    ██   ██ ██   ██ ██ ████   ██                    #
#                       ██    ██████  ███████ ██ ██ ██  ██                    #
#                       ██    ██   ██ ██   ██ ██ ██  ██ ██                    #
#                       ██    ██   ██ ██   ██ ██ ██   ████                    #
#                                                                             #
#=============================================================================#

#=============================================================================#
#                big_vision/trainers/proj/uvim/panoptic_task.py               #
#=============================================================================#

"""Inputs, outputs and losses for panoptic task."""
import big_vision.utils as u
import einops
import jax
import jax.numpy as jnp
import numpy as np

ONE_HOT_AXIS = -2


def input_pp(batch, config):
    """Make inputs for panoptic segmentation task."""
    if "labels" not in batch:
        # During predict of phase2 there is no 'labels' field.
        x = None
    else:
        hp, wp = config.model.patch_size
        x = {
            "semantics": batch["labels"][..., 0],
            "instances": batch["labels"][..., 1],
        }
        # Convert labels
        #   from (B, H, W)
        #   to (B, num_patches, num_classes, patch_size)
        for key in ["semantics", "instances"]:
            x[key] = jax.nn.one_hot(
                einops.rearrange(x[key],
                                 "b (hn hp) (wn wp) -> b (hn wn) (hp wp)",
                                 hp=hp,
                                 wp=wp),
                num_classes=config.model.inputs[key][ONE_HOT_AXIS],
                axis=ONE_HOT_AXIS)
    ctx = batch.get("image_ctx", batch.get("image", None))
    return {"ctx": ctx, "x": x}


def loss_fn(logits, batch, config):
    """Compute loss for panoptic task."""
    labels = input_pp(batch, config)["x"]
    losses = {}
    for key in ["semantics", "instances"]:
        losses[f"loss_{key}"] = u.softmax_xent(logits=logits[key],
                                               labels=labels[key],
                                               reduction=False,
                                               axis=ONE_HOT_AXIS)
    return sum(losses.values()), losses


def predict_outputs(logits, config, min_fraction=0.0):
    """Make outputs for panoptic segmentation task."""
    # Map logits to (height, width, channels).
    hp, wp = config.model.patch_size
    hn, wn = np.array(config.model.input_size) // np.array((hp, wp))
    outputs = {}
    for key in ["semantics", "instances"]:
        assert ONE_HOT_AXIS == -2, "Rearrange below depends on this."
        outputs[key] = einops.rearrange(
            logits[key],
            "b (hn wn) c (hp wp) -> b (hn hp) (wn wp) c",
            hn=hn,
            wn=wn,
            hp=hp,
            wp=wp)
    return panoptic_predictions_from_logits(**outputs,
                                            min_fraction=min_fraction)


def panoptic_predictions_from_logits(semantics, instances, min_fraction=0.0):
    """Make panoptic prediction from logits."""
    ins = jnp.argmax(instances, axis=-1)
    # Note: Make sure each instance has all pixels annotated with same label.
    # Otherwise they are further split into more instances and greatly affect
    # the number of unmatched predicted segments (FP) and RQ.
    masks = jax.nn.one_hot(ins, instances.shape[-1], dtype=jnp.int32)
    label = jnp.argmax(jnp.einsum("bhwk,bhwn->bnk", semantics, masks), axis=-1)
    sem = jnp.einsum("bhwn,bn->bhw", masks, label)
    out = jnp.stack([sem, ins], axis=-1)
    # Filter out small objects
    fraction = jnp.sum(masks, axis=(1, 2), keepdims=True) / np.prod(
        ins.shape[1:3])
    mask_big = (fraction > min_fraction).astype("int32")
    mask_big_spatial = jnp.sum(masks * mask_big, axis=-1, keepdims=True) > 0
    return out * mask_big_spatial.astype("int32")


#=============================================================================#
#                    big_vision/trainers/proj/uvim/vqvae.py                   #
#=============================================================================#
#====  STAGE I

"""Train loop for training the stage-I model."""
# pylint: disable=consider-using-from-import
import functools
import importlib
import multiprocessing.pool
import os

from absl import app
from absl import flags
from absl import logging
from big_vision import input_pipeline
import big_vision.datasets.core as ds_core
import big_vision.evaluators.common as eval_common
import big_vision.optax as bv_optax
import big_vision.pp.builder as pp_builder
import big_vision.utils as u
from clu import parameter_overview
import flax
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
import optax

import tensorflow.io.gfile as gfile

SG = jax.lax.stop_gradient
partial = functools.partial

config_flags.DEFINE_config_file("config",
                                None,
                                "Training configuration.",
                                lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup",
                     default=False,
                     help="Delete workdir (only) after successful completion.")

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()


def main(argv):
    del argv

    config = flags.FLAGS.config
    workdir = flags.FLAGS.workdir
    logging.info("Workdir: %s", workdir)

    logging.info(
        "\u001b[33mHello from process %i holding %i/%i devices and "
        "writing to workdir %s.\u001b[0m", jax.process_index(),
        jax.local_device_count(), jax.device_count(), workdir)

    # Define task input, loss and predict functions.
    task_module = importlib.import_module(f"big_vision.trainers.{config.task}")
    input_pp_fn = partial(task_module.input_pp, config=config)
    task_loss_fn = partial(task_module.loss_fn, config=config)
    predict_outputs_fn = partial(task_module.predict_outputs, config=config)

    save_ckpt_path = None
    if workdir:  # Always create if requested, even if we may not write into it.
        gfile.makedirs(workdir)
        save_ckpt_path = os.path.join(workdir, "checkpoint.npz")

    # The pool is used to perform misc operations such as logging in async way.
    pool = multiprocessing.pool.ThreadPool()

    # Here we register preprocessing ops from modules listed on `pp_modules`.
    for m in config.get("pp_modules",
                        ["ops_general", "ops_image", "proj.uvim.pp_ops"]):
        importlib.import_module(f"big_vision.pp.{m}")

    # This seed makes the Jax part of things (like model init) deterministic.
    # However, full training still won't be deterministic, for example due to the
    # tf.data pipeline not being deterministic even if we would set TF seed.
    # See (internal link) for a fun read on what it takes.
    rng = jax.random.PRNGKey(config.get("seed", 0))

    # These functions do more stuff internally, for OSS release we mock them by
    # trivial alternatives in order to minize disruptions in the code.
    xid, wid = -1, -1
    fillin = lambda s: s

    def info(s, *a):
        logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)

    def write_note(note):
        if jax.process_index() == 0:
            info("%s", note)

    write_note("Initializing...")

    batch_size = config.input.batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must "
            f"be divisible by device number ({jax.device_count()})")
    info(
        "Global batch size %d on %d hosts results in %d local batch size. With "
        "%d dev per host (%d dev total), that's a %d per-device batch size.",
        batch_size, jax.process_count(), batch_size // jax.process_count(),
        jax.local_device_count(), jax.device_count(),
        batch_size // jax.device_count())

    # First thing after above sanity checks, so we can log "start" ticks.
    mw = u.BigVisionMetricWriter(xid, wid, workdir, config)
    chrono = u.Chrono()

    write_note("Initializing train dataset...")
    train_data = ds_core.get(**config.input.data)
    train_ds = input_pipeline.make_for_train(
        data=train_data.get_tfdata(ordered=False),
        batch_size=batch_size,
        preprocess_fn=pp_builder.get_preprocess_fn(config.input.get("pp")),
        shuffle_buffer_size=config.input.get("shuffle_buffer_size"),
        cache_raw=config.input.get("cache_raw", False),
        filter_fn=config.input.get("filter_fn"),
    )

    # Start prefetching already.
    n_prefetch = config.get("prefetch_to_device", 1)
    train_iter = input_pipeline.start_input_pipeline(train_ds, n_prefetch)
    ntrain_img = train_data.total_examples

    def get_steps(name, default=ValueError):  # partial doesn't work well here.
        return u.steps(name, config, ntrain_img, batch_size, default)

    total_steps = get_steps("total")

    info("Running for %d steps, that means %f epochs", total_steps,
         total_steps * batch_size / ntrain_img)

    write_note(f"Initializing {config.model_name} model...")
    model_mod = importlib.import_module(
        f"big_vision.models.{config.model_name}")
    model = model_mod.Model(**config.model)

    # We want all parameters to be created in host RAM, not on any device, they'll
    # be sent there later as needed, otherwise we already encountered two
    # situations where we allocate them twice.
    @partial(jax.jit, backend="cpu")
    def init(rng):
        batch = jax.tree_map(
            lambda x: jnp.zeros(x.shape, x.dtype.as_numpy_dtype),
            train_ds.element_spec)
        init_res = flax.core.unfreeze(model.init(rng, **input_pp_fn(batch)))
        params, state = init_res["params"], init_res["state"]

        # Set bias in the heads to a low value, such that loss is small initially.
        for key in config.model.outputs:
            params[f"head_{key}"]["bias"] = jnp.full_like(
                params[f"head_{key}"]["bias"], config.get("init_head_bias", 0))

        return params, state

    rng, rng_init = jax.random.split(rng)

    rng_init_params, rng_init_state = jax.random.split(rng_init)
    params_cpu, state_cpu = init({
        "params": rng_init_params,
        "state": rng_init_state
    })

    if jax.process_index() == 0:
        num_params = sum(p.size for p in jax.tree_leaves(params_cpu))
        parameter_overview.log_parameter_overview(params_cpu,
                                                  msg="init params")
        mw.measure("num_params", num_params)

    write_note(f"Initializing {config.optax_name} optimizer...")
    tx, sched_fns = bv_optax.make(config,
                                  params_cpu,
                                  sched_kw=dict(total_steps=total_steps,
                                                batch_size=batch_size,
                                                data_size=ntrain_img))

    # We jit this, such that the arrays are created on the CPU, not device[0].
    opt_cpu = jax.jit(tx.init, backend="cpu")(params_cpu)
    sched_fns_cpu = [
        jax.jit(sched_fn, backend="cpu") for sched_fn in sched_fns
    ]

    @partial(jax.pmap,
             axis_name="batch",
             donate_argnums=(0, 1, 2),
             static_broadcasted_argnums=(5, ))
    def update_fn(params, opt, state, batch, rng, update_dict=True):
        """Update step."""
        measurements = {}

        # Get device-specific loss rng.
        rng, rng_model = jax.random.split(rng, 2)
        rng_model_local = jax.random.fold_in(rng_model,
                                             jax.lax.axis_index("batch"))

        def loss_fn(params, state, batch):
            (logits,
             out), mutated_col = model.apply({
                 "params": params,
                 "state": state
             },
                                             **input_pp_fn(batch),
                                             train=True,
                                             update_dict=update_dict,
                                             rngs={
                                                 "dropout": rng_model_local,
                                                 "vqvae": rng_model
                                             },
                                             mutable=["state"])
            btlneck = out["bottleneck"]
            btlneck_q = out["bottleneck_q"]

            loss_rec, logs = jax.tree_map(jnp.mean,
                                          task_loss_fn(logits, batch))
            loss_commitment = jnp.mean(jnp.square(btlneck - SG(btlneck_q)))
            loss = loss_rec + config.get("w_commitment",
                                         0.25) * loss_commitment
            aux = {
                "loss_rec":
                jax.lax.pmean(loss_rec, axis_name="batch"),
                "loss_commitment":
                jax.lax.pmean(loss_commitment, axis_name="batch"),
                "codebook_zeros_ratio":
                out["codebook_zeros_ratio"],
                "codebook_max_ratio":
                out["codebook_max_ratio"],
                "state":
                mutated_col["state"],
                **jax.tree_map(partial(jax.lax.pmean, axis_name="batch"), logs),
            }
            return loss, aux

        (l, aux), grads = jax.value_and_grad(loss_fn,
                                             has_aux=True)(params, state,
                                                           batch)
        l, grads = jax.lax.pmean((l, grads), axis_name="batch")
        updates, opt = tx.update(grads, opt, params)
        params = optax.apply_updates(params, updates)
        state = aux.pop("state")
        measurements = {**measurements, **aux}

        gs = jax.tree_leaves(
            bv_optax.replace_frozen(config.schedule, grads, 0.))
        measurements["l2_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
        ps = jax.tree_leaves(params)
        measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
        us = jax.tree_leaves(updates)
        measurements["l2_updates"] = jnp.sqrt(sum([jnp.vdot(u, u)
                                                   for u in us]))

        return params, opt, state, l, rng, measurements

    # Define evaluators.
    def validation_fn(params, batch):
        """Compute per-example metrics."""
        logits, out = model.apply(params, **input_pp_fn(batch))
        _, losses = task_loss_fn(logits, batch)
        btlneck = out["bottleneck"]
        btlneck_q = out["bottleneck_q"]
        losses["loss_commitment"] = jnp.square(btlneck - btlneck_q)
        return jax.tree_map(
            lambda x: jnp.mean(x, axis=tuple(range(1, x.ndim))), losses)

    def predict_fn(params, batch):
        logits, _ = model.apply(params, **input_pp_fn(batch))
        outputs = predict_outputs_fn(logits)
        return outputs

    # Only initialize evaluators when they are first needed.
    @functools.lru_cache(maxsize=None)
    def evaluators():
        return eval_common.from_config(
            config, {
                "predict": predict_fn,
                "validation": validation_fn
            }, lambda s: write_note(
                f"Initializing evaluator: {s}...\n{chrono.note}"))

    # Decide how to initialize training. The order is important.
    # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
    # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
    # 3. Initialize model from something, e,g, start a fine-tuning job.
    # 4. Train from scratch.
    resume_ckpt_path = None
    if save_ckpt_path and gfile.exists(save_ckpt_path):
        resume_ckpt_path = save_ckpt_path
    elif config.get("resume"):
        resume_ckpt_path = fillin(config.resume)
    if resume_ckpt_path:
        write_note("Resume training from checkpoint...")
        checkpoint = {
            "params": params_cpu,
            "state": state_cpu,
            "opt": opt_cpu,
            "chrono": chrono.save(),
        }
        checkpoint_tree = jax.tree_structure(checkpoint)
        loaded = u.load_checkpoint(checkpoint_tree, resume_ckpt_path)
        # bfloat16 type gets lost when data is saved to disk, so we recover it.
        checkpoint = jax.tree_map(u.recover_dtype, loaded)
        params_cpu = checkpoint["params"]
        state_cpu = checkpoint["state"]
        opt_cpu = checkpoint["opt"]
        chrono.load(checkpoint["chrono"])
    elif config.get("model_init"):
        write_note(f"Initialize model from {config.model_init}...")
        params_cpu, state_cpu = model_mod.load(
            {
                "params": params_cpu,
                "state": state_cpu
            }, config.model_init, config.model, **config.get("model_load", {}))
        if jax.process_index() == 0:
            parameter_overview.log_parameter_overview(params_cpu,
                                                      msg="restored params")

    write_note("Kicking off misc stuff...")
    first_step = bv_optax.get_count(opt_cpu)
    chrono.inform(first_step, total_steps, batch_size, ntrain_img / batch_size)
    prof = None  # Keeps track of start/stop of profiler state.

    write_note(f"Replicating...\n{chrono.note}")
    params_repl = flax.jax_utils.replicate(params_cpu)
    opt_repl = flax.jax_utils.replicate(opt_cpu)
    state_repl = flax.jax_utils.replicate(state_cpu)

    rng, rng_loop = jax.random.split(rng, 2)
    rngs_loop = flax.jax_utils.replicate(rng_loop)
    ckpt_writer = None

    write_note(f"First step compilations...\n{chrono.note}")
    error = None  # For exiting with an error after cleanup. Avoids indentation.

    # Using a python integer for step here, because opt.state.step is allocated
    # on TPU during replication.
    for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
        mw.step_start(step)

        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            params_repl, opt_repl, state_repl, loss_value, rngs_loop, measurements = (
                update_fn(params_repl, opt_repl, state_repl, batch, rngs_loop,
                          not config.get("freeze_dict", True)))

        # On the first host, let's always profile a handful of early steps.
        if jax.process_index() == 0:
            prof = u.startstop_prof(prof, step, first_step,
                                    get_steps("log_training"))

        # Report training progress
        if (u.itstime(step, get_steps("log_training"), total_steps, host=0)
                or chrono.warmup and jax.process_index() == 0):
            for i, sched_fn_cpu in enumerate(sched_fns_cpu):
                mw.measure(f"global_schedule{i if i else ''}",
                           sched_fn_cpu(step - 1))
            l = mw.measure("training_loss", loss_value[0])
            for name, value in measurements.items():
                mw.measure(name, value[0])
            chrono.tick(step, mw.measure, write_note)
            if not np.isfinite(l):
                error = (f"The loss became nan or inf somewhere within steps "
                         f"[{step - get_steps('log_training')}, {step}]")
                break

        # Checkpoint saving
        if (save_ckpt_path and
            (u.itstime(step, get_steps("ckpt", None), total_steps, host=0)
             or u.itstime(
                 step, get_steps("keep_ckpt", None), total_steps, host=0))):
            chrono.pause(wait_for=(params_repl, opt_repl, state_repl))
            u.checkpointing_timeout(ckpt_writer, config.get("ckpt_timeout", 1))
            # We need to transfer the weights over now or else we risk keeping them
            # alive while they'll be updated in a future step, creating hard to debug
            # memory errors (see (internal link)). Also, takes device 0's params only.
            params_cpu, opt_cpu, state_cpu = jax.tree_map(
                lambda x: np.array(x[0]), (params_repl, opt_repl, state_repl))

            # Check whether we want to keep a copy of the current checkpoint.
            copy_step = None
            if u.itstime(step, get_steps("keep_ckpt", None), total_steps):
                copy_step = step

            ckpt = {
                "params": params_cpu,
                "state": state_cpu,
                "opt": opt_cpu,
                "chrono": chrono.save(),
            }
            ckpt_writer = pool.apply_async(u.save_checkpoint,
                                           (ckpt, save_ckpt_path, copy_step))
            chrono.resume()

        for (name, evaluator, log_steps, prefix) in evaluators():
            if u.itstime(step, log_steps, total_steps):
                chrono.pause(wait_for=(params_repl, state_repl))
                write_note(f"{name} evaluation...\n{chrono.note}")
                for key, value in evaluator.run({
                        "params": params_repl,
                        "state": state_repl
                }):
                    mw.measure(f"{prefix}{key}", value)
                chrono.resume()
        mw.step_end()

    # Always give a chance to stop the profiler, no matter how things ended.
    # TODO: can we also do this when dying of an exception like OOM?
    if jax.process_index() == 0 and prof is not None:
        u.startstop_prof(prof)

    # Support eval only runs: run evaluation if total_steps (or num_epochs) is 0.
    if total_steps == 0:
        for (name, evaluator, _, prefix) in evaluators():
            write_note(f"{name} evaluation...\n{chrono.note}")
            for key, value in evaluator.run({
                    "params": params_repl,
                    "state": state_repl
            }):
                mw.measure(f"{prefix}{key}", value)

    # Last note needs to happen before the pool's closed =)
    if not error:
        write_note(f"Done!\n{chrono.note}")
    else:
        write_note(f"Failed!\n{error}\n{chrono.note}")

    pool.close()
    pool.join()
    mw.close()

    # Make sure all hosts stay up until the end of main.
    u.sync()

    # Before cleanup, as cleanup should only run for successful jobs.
    if error is not None:
        raise RuntimeError(error)

    u.maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, info)


if __name__ == "__main__":
    app.run(main)



#=============================================================================#
#                    big_vision/trainers/proj/uvim/train.py                   #
#=============================================================================#
#====  STAGE II

"""Train loop for training the stage-II model."""
# pylint: disable=consider-using-from-import
import functools
import importlib
import multiprocessing.pool
import os

from absl import app
from absl import flags
from absl import logging
from big_vision import input_pipeline
import big_vision.datasets.core as ds_core
import big_vision.evaluators.common as eval_common
import big_vision.models.proj.uvim.decode as decode
import big_vision.optax as bv_optax
import big_vision.pp.builder as pp_builder
import big_vision.utils as u
from clu import parameter_overview
import flax
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
import optax

import tensorflow.io.gfile as gfile

config_flags.DEFINE_config_file("config",
                                None,
                                "Training configuration.",
                                lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup",
                     default=False,
                     help="Delete workdir (only) after successful completion.")

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS
ONE_HOT_AXIS = -2
partial = functools.partial


def get_model(config):
    mod = importlib.import_module(f"big_vision.models.{config.model_name}")
    model = mod.Model(**config.model)
    return model, mod


def setup_task(config):
    """Get functions and params to encode and decode labels as token sequences."""
    config = config.oracle

    # Define task input and predict functions.
    task_module = importlib.import_module(f"big_vision.trainers.{config.task}")
    input_fn = partial(task_module.input_pp, config=config)
    predict_outputs_fn = partial(task_module.predict_outputs, config=config)

    oracle, mod = get_model(config)
    if config.get("model_init", None):
        params, state = mod.load(None, config.model_init)
        params = {"params": params, "state": state}
    else:
        params = {}

    def encode_labels(params, batch):
        inputs = input_fn(batch)
        code = oracle.apply(params, **inputs, method=oracle.encode)[1]["code"]
        return code + 1  # To avoid padding symbol.

    def decode_labels(params, code, batch, **kwargs):
        code = code - 1
        inputs = input_fn(batch)
        inputs["x"] = code
        logits, _ = oracle.apply(params,
                                 **inputs,
                                 discrete_input=True,
                                 **kwargs,
                                 method=oracle.decode)
        return logits

    return encode_labels, decode_labels, predict_outputs_fn, params


def main(argv):
    del argv

    config = FLAGS.config
    workdir = FLAGS.workdir
    logging.info(
        "\u001b[33mHello from process %i holding %i/%i devices and "
        "writing to workdir %s.\u001b[0m", jax.process_index(),
        jax.local_device_count(), jax.device_count(), workdir)

    save_ckpt_path = None
    if workdir:  # Always create if requested, even if we may not write into it.
        gfile.makedirs(workdir)
        save_ckpt_path = os.path.join(workdir, "checkpoint.npz")

    # The pool is used to perform misc operations such as logging in async way.
    pool = multiprocessing.pool.ThreadPool()

    # Here we register preprocessing ops from modules listed on `pp_modules`.
    for m in config.get("pp_modules",
                        ["ops_general", "ops_image", "proj.uvim.pp_ops"]):
        importlib.import_module(f"big_vision.pp.{m}")

    # This seed makes the Jax part of things (like model init) deterministic.
    # However, full training still won't be deterministic, for example due to the
    # tf.data pipeline not being deterministic even if we would set TF seed.
    # See (internal link) for a fun read on what it takes.
    rng = jax.random.PRNGKey(config.get("seed", 0))

    # These functions do more stuff internally, for OSS release we mock them by
    # trivial alternatives in order to minize disruptions in the code.
    xid, wid = -1, -1
    fillin = lambda s: s

    def info(s, *a):
        logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)

    def write_note(note):
        if jax.process_index() == 0:
            info("%s", note)

    write_note("Initializing...")

    batch_size = config.input.batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must "
            f"be divisible by device number ({jax.device_count()})")
    info(
        "Global batch size %d on %d hosts results in %d local batch size. With "
        "%d dev per host (%d dev total), that's a %d per-device batch size.",
        batch_size, jax.process_count(), batch_size // jax.process_count(),
        jax.local_device_count(), jax.device_count(),
        batch_size // jax.device_count())

    # First thing after above sanity checks, so we can log "start" ticks.
    mw = u.BigVisionMetricWriter(xid, wid, workdir, config)
    chrono = u.Chrono()

    write_note("Initializing train dataset...")
    train_data = ds_core.get(**config.input.data)
    train_ds = input_pipeline.make_for_train(
        data=train_data.get_tfdata(ordered=False),
        batch_size=batch_size,
        preprocess_fn=pp_builder.get_preprocess_fn(config.input.get("pp")),
        shuffle_buffer_size=config.input.get("shuffle_buffer_size"),
        cache_raw=config.input.get("cache_raw", False),
        filter_fn=config.input.get("filter_fn"),
    )

    # Start prefetching already.
    n_prefetch = config.get("prefetch_to_device", 1)
    train_iter = input_pipeline.start_input_pipeline(train_ds, n_prefetch)
    ntrain_img = train_data.total_examples

    def get_steps(name, default=ValueError):  # partial doesn't work well here.
        return u.steps(name, config, ntrain_img, batch_size, default)

    total_steps = get_steps("total")

    info("Running for %d steps, that means %f epochs", total_steps,
         total_steps * batch_size / ntrain_img)

    write_note(f"Initializing {config.model_name} model...")
    model, model_mod = get_model(config)

    encode_labels, decode_labels, predict_outputs_fn, task_params = (
        setup_task(config))

    # We want all parameters to be created in host RAM, not on any device, they'll
    # be sent there later as needed, otherwise we already encountered two
    # situations where we allocate them twice.
    @partial(jax.jit, backend="cpu")
    def init(rng):
        batch = jax.tree_map(
            lambda x: jnp.zeros(x.shape, x.dtype.as_numpy_dtype),
            train_ds.element_spec)
        images = batch["image"]
        labels = encode_labels(task_params, batch)
        variables = model.init(rng, images, labels)
        params = flax.core.unfreeze(variables["params"])
        return params

    rng, init_rng = jax.random.split(rng)
    params_cpu = init(init_rng)

    if jax.process_index() == 0:
        num_params = sum(p.size for p in jax.tree_leaves(params_cpu))
        parameter_overview.log_parameter_overview(params_cpu,
                                                  msg="init params")
        mw.measure("num_params", num_params)

    write_note(f"Initializing {config.optax_name} optimizer...")
    tx, sched_fns = bv_optax.make(config,
                                  params_cpu,
                                  sched_kw=dict(total_steps=total_steps,
                                                batch_size=batch_size,
                                                data_size=ntrain_img))

    # We jit this, such that the arrays are created on the CPU, not device[0].
    opt_cpu = jax.jit(tx.init, backend="cpu")(params_cpu)
    sched_fns_cpu = [
        jax.jit(sched_fn, backend="cpu") for sched_fn in sched_fns
    ]

    @partial(jax.pmap, axis_name="batch", donate_argnums=(0, 1))
    def update_fn(params, opt, batch, update_rng, task_params):
        """Update step."""
        images = batch["image"]
        labels = encode_labels(task_params, batch)

        measurements = {}

        rng, new_rng = jax.random.split(update_rng)
        # bind the rng key to the device id (which is unique across hosts)
        rng_local = jax.random.fold_in(rng, jax.lax.axis_index("batch"))

        def loss_fn(params, images, labels):
            logits = model.apply({"params": params},
                                 images,
                                 labels,
                                 train=True,
                                 rngs={"dropout": rng_local})
            loss = u.weighted_softmax_xent(logits=logits,
                                           labels=labels,
                                           reduction=True,
                                           normalize=True)
            return loss

        l, grads = jax.value_and_grad(loss_fn)(params, images, labels)
        l, grads = jax.lax.pmean((l, grads), axis_name="batch")
        updates, opt = tx.update(grads, opt, params)
        params = optax.apply_updates(params, updates)

        gs = jax.tree_leaves(
            bv_optax.replace_frozen(config.schedule, grads, 0.))
        measurements["l2_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
        ps = jax.tree_leaves(params)
        measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
        us = jax.tree_leaves(updates)
        measurements["l2_updates"] = jnp.sqrt(sum([jnp.vdot(u, u)
                                                   for u in us]))

        return params, opt, l, new_rng, measurements

    # Define evaluators.
    def validation_fn(params, batch):
        """Compute per-example metrics."""
        params, task_params = params["params"], params["task_params"]
        images = batch["image"]
        labels = encode_labels(task_params, batch)
        logits = model.apply({"params": params}, images, labels, train=False)
        loss = u.weighted_softmax_xent(logits=logits,
                                       labels=labels,
                                       reduction=False,
                                       normalize=True)
        losses = {"loss": loss}
        return jax.tree_map(
            lambda x: jnp.mean(x, axis=tuple(range(1, x.ndim))), losses)

    def predict_fn(params, batch, seed=0, temperature=1e-7, **extra):
        params, task_params = params["params"], params["task_params"]

        # Derive a rng key from the inputs so that all batches use different keys.
        if "image/id" in batch:
            key = batch["image/id"]
        else:
            key = batch["image"].sum(axis=[1, 2, 3]).astype(jnp.int32)
        local_rng = jax.lax.scan(
            lambda k, x: (jax.random.fold_in(k, x), None),
            jax.random.PRNGKey(seed),
            key,
        )[0]

        images = batch["image"]
        batch_size = images.shape[0]
        prompts = jnp.zeros((batch_size, config.model.seq_len),
                            dtype=jnp.int32)
        seqs, _, _ = decode.temperature_sampling(params={"params": params},
                                                 model=model,
                                                 seed=local_rng,
                                                 inputs=images,
                                                 prompts=prompts,
                                                 num_samples=1,
                                                 eos_token=-1,
                                                 prefill=False,
                                                 temperature=temperature)
        seqs = jnp.squeeze(seqs, 1)
        logits = decode_labels(task_params, seqs, batch)
        return predict_outputs_fn(logits, **extra)

    # Only initialize evaluators when they are first needed.
    @functools.lru_cache(maxsize=None)
    def evaluators():
        return eval_common.from_config(
            config, {
                "predict": predict_fn,
                "validation": validation_fn
            }, lambda s: write_note(
                f"Initializing evaluator: {s}...\n{chrono.note}"))

    # Decide how to initialize training. The order is important.
    # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
    # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
    # 3. Initialize model from something, e,g, start a fine-tuning job.
    # 4. Initialize part of the model from something, eg. only encoder or decoder.
    # 5. Train from scratch.
    resume_ckpt_path = None
    if save_ckpt_path and gfile.exists(save_ckpt_path):
        resume_ckpt_path = save_ckpt_path
    elif config.get("resume"):
        resume_ckpt_path = fillin(config.resume)
    if resume_ckpt_path:
        write_note("Resume training from checkpoint...")
        checkpoint = {
            "params": params_cpu,
            "opt": opt_cpu,
            "chrono": chrono.save(),
        }
        checkpoint_tree = jax.tree_structure(checkpoint)
        loaded = u.load_checkpoint(checkpoint_tree, resume_ckpt_path)
        # bfloat16 type gets lost when data is saved to disk, so we recover it.
        checkpoint = jax.tree_map(u.recover_dtype, loaded)
        params_cpu, opt_cpu = checkpoint["params"], checkpoint["opt"]
        chrono.load(checkpoint["chrono"])
    elif config.get("model_init"):
        write_note(f"Initialize model from {config.model_init}...")
        params_cpu = model_mod.load(params_cpu,
                                    config.model_init, config.model,
                                    **config.get("model_load", {}))
        if jax.process_index() == 0:
            parameter_overview.log_parameter_overview(params_cpu,
                                                      msg="restored params")

    write_note("Kicking off misc stuff...")
    first_step = bv_optax.get_count(opt_cpu)
    chrono.inform(first_step, total_steps, batch_size, ntrain_img / batch_size)
    prof = None  # Keeps track of start/stop of profiler state.

    write_note(f"Replicating...\n{chrono.note}")
    params_repl = flax.jax_utils.replicate(params_cpu)
    opt_repl = flax.jax_utils.replicate(opt_cpu)
    task_params = flax.jax_utils.replicate(task_params)
    update_rngs = flax.jax_utils.replicate(rng)

    ckpt_writer = None

    write_note(f"First step compilations...\n{chrono.note}")
    error = None  # For exiting with an error after cleanup. Avoids indentation.

    # Using a python integer for step here, because opt.state.step is allocated
    # on TPU during replication.
    for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
        mw.step_start(step)

        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            params_repl, opt_repl, loss_value, update_rngs, measurements = (
                update_fn(params_repl,
                          opt_repl,
                          batch,
                          update_rng=update_rngs,
                          task_params=task_params))

        # On the first host, let's always profile a handful of early steps.
        if jax.process_index() == 0:
            prof = u.startstop_prof(prof, step, first_step,
                                    get_steps("log_training"))

        # Report training progress
        if (u.itstime(step, get_steps("log_training"), total_steps, host=0)
                or chrono.warmup and jax.process_index() == 0):
            for i, sched_fn_cpu in enumerate(sched_fns_cpu):
                mw.measure(f"global_schedule{i if i else ''}",
                           sched_fn_cpu(step - 1))
            l = mw.measure("training_loss", loss_value[0])
            for name, value in measurements.items():
                mw.measure(name, value[0])
            chrono.tick(step, mw.measure, write_note)
            if not np.isfinite(l):
                error = (f"The loss became nan or inf somewhere within steps "
                         f"[{step - get_steps('log_training')}, {step}]")
                break

        # Checkpoint saving
        if (save_ckpt_path and
            (u.itstime(step, get_steps("ckpt", None), total_steps, host=0)
             or u.itstime(
                 step, get_steps("keep_ckpt", None), total_steps, host=0))):
            chrono.pause(wait_for=(params_repl, opt_repl))
            u.checkpointing_timeout(ckpt_writer, config.get("ckpt_timeout", 1))
            # We need to transfer the weights over now or else we risk keeping them
            # alive while they'll be updated in a future step, creating hard to debug
            # memory errors (see (internal link)). Also, takes device 0's params only.
            opt_cpu = jax.tree_map(lambda x: np.array(x[0]), opt_repl)
            params_cpu = jax.tree_map(lambda x: np.array(x[0]), params_repl)

            # Check whether we want to keep a copy of the current checkpoint.
            copy_step = None
            if u.itstime(step, get_steps("keep_ckpt", None), total_steps):
                copy_step = step

            ckpt = {
                "params": params_cpu,
                "opt": opt_cpu,
                "chrono": chrono.save()
            }
            ckpt_writer = pool.apply_async(u.save_checkpoint,
                                           (ckpt, save_ckpt_path, copy_step))
            chrono.resume()

        for (name, evaluator, log_steps, prefix) in evaluators():
            if u.itstime(step,
                         log_steps,
                         total_steps,
                         first=log_steps < total_steps,
                         last=False):
                chrono.pause(wait_for=(params_repl, task_params))
                write_note(f"{name} evaluation...\n{chrono.note}")
                for key, value in evaluator.run({
                        "params": params_repl,
                        "task_params": task_params
                }):
                    mw.measure(f"{prefix}{key}", value)
                chrono.resume()
        mw.step_end()

    # Always give a chance to stop the profiler, no matter how things ended.
    # TODO: can we also do this when dying of an exception like OOM?
    if jax.process_index() == 0 and prof is not None:
        u.startstop_prof(prof)

    # Run final evalution, also used for eval only jobs (when total_steps == 0).
    for (name, evaluator, _, prefix) in evaluators():
        write_note(f"{name} evaluation...\n{chrono.note}")
        for key, value in evaluator.run({
                "params": params_repl,
                "task_params": task_params
        }):
            mw.measure(f"{prefix}{key}", value)

    # Last note needs to happen before the pool's closed =)
    if not error:
        write_note(f"Done!\n{chrono.note}")
    else:
        write_note(f"Failed!\n{error}\n{chrono.note}")

    pool.close()
    pool.join()
    mw.close()

    # Make sure all hosts stay up until the end of main.
    u.sync()

    # Before cleanup, as cleanup should only run for successful jobs.
    if error is not None:
        raise RuntimeError(error)

    u.maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, info)


if __name__ == "__main__":
    app.run(main)
