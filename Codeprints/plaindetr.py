#=============================================================================#
#         ____    _           _           ____    _____   _____   ____        #
#        |  _ \  | |   __ _  (_)  _ __   |  _ \  | ____| |_   _| |  _ \       #
#        | |_) | | |  / _` | | | | '_ \  | | | | |  _|     | |   | |_) |      #
#        |  __/  | | | (_| | | | | | | | | |_| | | |___    | |   |  _ <       #
#        |_|     |_|  \__,_| |_| |_| |_| |____/  |_____|   |_|   |_| \_\      #
#                                                                             #
#=============================================================================#
## https://github.com/impiga/Plain-DETR
## Ref: 6ad930b
#
#
# Plain-DETR/
# ├── configs/
# │   ├── swinv2_small_mim_pt_boxrpe_reparam.sh*
# │   ├── swinv2_small_mim_pt_boxrpe.sh*
# │   ├── swinv2_small_sup_pt_ape.sh*
# │   └── swinv2_small_sup_pt_boxrpe.sh*
# ├── datasets/
# │   ├── torchvision_datasets/
# │   │   └── coco.py
# │   ├── coco_eval.py
# │   ├── coco_panoptic.py
# │   ├── coco.py
# │   ├── data_prefetcher.py
# │   ├── panoptic_eval.py
# │   ├── samplers.py
# │   └── transforms.py
# ├── models/
# │   ├── backbone.py
# │   ├── detr.py
# │   ├── global_ape_decoder.py
# │   ├── global_rpe_decomp_decoder.py
# │   ├── matcher.py
# │   ├── position_encoding.py
# │   ├── segmentation.py
# │   ├── swin_transformer_v2.py
# │   ├── transformer.py
# │   └── utils.py
# ├── tools/
# │   ├── launch.py
# │   ├── prepare_pt_model.sh
# │   ├── run_dist_launch.sh
# │   └── run_dist_slurm.sh
# ├── util/
# │   ├── box_ops.py
# │   ├── misc.py
# │   └── plot_utils.py
# ├── benchmark.py
# ├── engine.py
# └── main.py

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
#                                models/detr.py                               #
#=============================================================================#

#$#>START: models/detr.py

import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
    inverse_sigmoid,
    _get_clones,
)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (
    DETRsegm,
    PostProcessPanoptic,
    PostProcessSegm,
    dice_loss,
    sigmoid_focal_loss,
)
from .transformer import build_transformer
import copy


class PlainDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_feature_levels,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        num_queries_one2one=300,
        num_queries_one2many=0,
        mixed_selection=False,
    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture.
                See transformer.py
            num_classes: number of object classes
            aux_loss: True if auxiliary decoding losses (loss at each decoder
                layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            num_queries_one2one: number of object queries for one-to-one
                matching part
            num_queries_one2many: number of object queries for one-to-many
                matching part
            mixed_selection: a trick for Deformable DETR two stage

        """
        super().__init__()
        num_queries = num_queries_one2one + num_queries_one2many
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        elif mixed_selection:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )
        ])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region
        # proposal generation
        num_pred = ((transformer.decoder.num_layers +
                     1) if two_stage else transformer.decoder.num_layers)
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:],
                              -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList(
                [self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        self.num_queries_one2one = num_queries_one2one
        self.mixed_selection = mixed_selection

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W],
                 containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits":
               the classification logits (including no-object) for all queries.
               Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes":
               The normalized boxes coordinates for all queries, represented as
               (center_x, center_y, height, width). These values are normalized
                in [0, 1],
               relative to the size of each individual image
                (disregarding possible padding).
               See PostProcess for information on how to retrieve the
                 unnormalized bounding box.
           - "aux_outputs":
               Optional, only returned when auxilary losses are activated.
               It is a list of
               dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        query_embeds = None
        if not self.two_stage or self.mixed_selection:
            query_embeds = self.query_embed.weight[0:self.num_queries, :]

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = (torch.zeros([
            self.num_queries,
            self.num_queries,
        ]).bool().to(src.device))
        self_attn_mask[
            self.num_queries_one2one:,
            0:self.num_queries_one2one,
        ] = True
        self_attn_mask[
            0:self.num_queries_one2one,
            self.num_queries_one2one:,
        ] = True

        (hs, init_reference, inter_references, enc_outputs_class,
         enc_outputs_coord_unact, enc_outputs_delta, output_proposals,
         max_shape) = self.transformer(srcs, masks, pos, query_embeds,
                                       self_attn_mask)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes_one2one.append(
                outputs_class[:, 0:self.num_queries_one2one])
            outputs_classes_one2many.append(
                outputs_class[:, self.num_queries_one2one:])

            outputs_coords_one2one.append(
                outputs_coord[:, 0:self.num_queries_one2one])
            outputs_coords_one2many.append(
                outputs_coord[:, self.num_queries_one2one:])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_classes_one2one,
                                                    outputs_coords_one2one)
            out["aux_outputs_one2many"] = self._set_aux_loss(
                outputs_classes_one2many, outputs_coords_one2many)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{
            "pred_logits": a,
            "pred_boxes": b
        } for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class PlainDETRReParam(PlainDETR):

    def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
        - samples.mask: a binary mask of shape [batch_size x H x W],
          containing 1 on padded pixels

        It returns a dict with the following elements:
        - "pred_logits": the classification logits (including no-object)
            for all queries.
                         Shape= [batch_size x num_queries x (num_classes + 1)]
        - "pred_boxes": The normalized boxes coordinates for all queries,
            represented as
                        (center_x, center_y, height, width).
                        These values are normalized in [0, 1], relative to the
                        size of each individual image (disregarding possible padding).
                        See PostProcess for information on how to retrieve the
                        unnormalized bounding box.
        - "aux_outputs": Optional, only returned when auxilary losses are activated.
                        It is a list of dictionnaries containing the two
                        above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        query_embeds = None
        if not self.two_stage or self.mixed_selection:
            query_embeds = self.query_embed.weight[0:self.num_queries, :]

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = (torch.zeros([
            self.num_queries,
            self.num_queries,
        ]).bool().to(src.device))
        self_attn_mask[
            self.num_queries_one2one:,
            0:self.num_queries_one2one,
        ] = True
        self_attn_mask[
            0:self.num_queries_one2one,
            self.num_queries_one2one:,
        ] = True

        (hs, init_reference, inter_references, enc_outputs_class,
         enc_outputs_coord_unact, enc_outputs_delta, output_proposals,
         max_shape) = self.transformer(srcs, masks, pos, query_embeds,
                                       self_attn_mask)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []

        outputs_coords_old_one2one = []
        outputs_deltas_one2one = []
        outputs_coords_old_one2many = []
        outputs_deltas_one2many = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                outputs_coord = box_ops.box_xyxy_to_cxcywh(
                    box_ops.delta2bbox(reference, tmp, max_shape))
            else:
                raise NotImplementedError

            outputs_classes_one2one.append(
                outputs_class[:, 0:self.num_queries_one2one])
            outputs_classes_one2many.append(
                outputs_class[:, self.num_queries_one2one:])

            outputs_coords_one2one.append(
                outputs_coord[:, 0:self.num_queries_one2one])
            outputs_coords_one2many.append(
                outputs_coord[:, self.num_queries_one2one:])

            outputs_coords_old_one2one.append(
                reference[:, :self.num_queries_one2one])
            outputs_coords_old_one2many.append(
                reference[:, self.num_queries_one2one:])
            outputs_deltas_one2one.append(tmp[:, :self.num_queries_one2one])
            outputs_deltas_one2many.append(tmp[:, self.num_queries_one2one:])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
            "pred_boxes_old": outputs_coords_old_one2one[-1],
            "pred_deltas": outputs_deltas_one2one[-1],
            "pred_boxes_old_one2many": outputs_coords_old_one2many[-1],
            "pred_deltas_one2many": outputs_deltas_one2many[-1],
        }

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_classes_one2one, outputs_coords_one2one,
                outputs_coords_old_one2one, outputs_deltas_one2one)
            out["aux_outputs_one2many"] = self._set_aux_loss(
                outputs_classes_one2many, outputs_coords_one2many,
                outputs_coords_old_one2many, outputs_deltas_one2many)

        if self.two_stage:
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord_unact,
                "pred_boxes_old": output_proposals,
                "pred_deltas": enc_outputs_delta,
            }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_coord_old,
                      outputs_deltas):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{
            "pred_logits": a,
            "pred_boxes": b,
            "pred_boxes_old": c,
            "pred_deltas": d,
        } for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1],
                                outputs_coord_old[:-1], outputs_deltas[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
    1) we compute hungarian assignment between ground truth boxes and the outputs of the model
    2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self,
                 num_classes,
                 matcher,
                 weight_dict,
                 losses,
                 focal_alpha=0.25,
                 reparam=False):
        """ Create the criterion.
        Parameters:
        num_classes: number of object categories, omitting the special no-object category
        matcher: module able to compute a matching between targets and proposals
        weight_dict: dict containing as key the names of the losses and as values
            their relative weight.
        losses: list of all the losses to be applied. See get_loss for list of
            available losses.
        focal_alpha: alpha in Focal Loss
        loss_bbox_type: how to perform loss_bbox
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.loss_bbox_type = 'l1' if (not reparam) else 'reparam'

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of
        dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [
                src_logits.shape[0], src_logits.shape[1],
                src_logits.shape[2] + 1
            ],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes,
            alpha=self.focal_alpha,
            gamma=2,
        ) * src_logits.shape[1])
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx],
                                                   target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number
        of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only.
        It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets],
                                      device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1)
                     != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes,
        the L1 regression loss and the GIoU loss targets dicts must contain the
        key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w),
           normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if self.loss_bbox_type == "l1":
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        elif self.loss_bbox_type == "reparam":
            src_deltas = outputs["pred_deltas"][idx]
            src_boxes_old = outputs["pred_boxes_old"][idx]
            target_deltas = box_ops.bbox2delta(src_boxes_old, target_boxes)
            loss_bbox = F.l1_loss(src_deltas, target_deltas, reduction="none")
        else:
            raise NotImplementedError

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            ))
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(
            [t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks,
                                            num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model
             for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses
                      applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for
        # normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes],
                                    dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes,
                              **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of
        # each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute,
                        # we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs["log"] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices,
                                           num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs["log"] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices,
                                       num_boxes, **kwargs)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, topk=100, reparam=False):
        super().__init__()
        self.topk = topk
        self.reparam = reparam
        print("topk for eval:", self.topk)

    @torch.no_grad()
    def forward(self, outputs, target_sizes, original_target_sizes=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model target_sizes: tensor of
            dimension [batch_size x 2] containing the size of each images of
            the batch
                          For evaluation, this must be the original image size
                          (before any data augmentation) For visualization,
                          this should be the image size after data augment,
                          but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        assert not self.reparam or original_target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(
            out_logits.shape[0], -1),
                                               self.topk,
                                               dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1,
                             topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        if self.reparam:
            img_h, img_w = img_h[:, None, None], img_w[:, None,
                                                       None]  # [BS, 1, 1]
            boxes[..., 0::2].clamp_(min=torch.zeros_like(img_w), max=img_w)
            boxes[..., 1::2].clamp_(min=torch.zeros_like(img_h), max=img_h)
            scale_h, scale_w = (original_target_sizes / target_sizes).unbind(1)
            scale_fct = torch.stack([scale_w, scale_h, scale_w, scale_h],
                                    dim=1)
        else:
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{
            "scores": s,
            "labels": l,
            "boxes": b
        } for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 20 if args.dataset_file != "coco" else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model_class = PlainDETR if (not args.reparam) else PlainDETRReParam
    model = model_class(
        backbone,
        transformer,
        num_classes=num_classes,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        num_queries_one2one=args.num_queries_one2one,
        num_queries_one2many=args.num_queries_one2many,
        mixed_selection=args.mixed_selection,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {
        "loss_ce": args.cls_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef
    }
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({
                k + f"_{i}": v
                for k, v in weight_dict.items()
            })
        aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    new_dict = dict()
    for key, value in weight_dict.items():
        new_dict[key] = value
        new_dict[key + "_one2many"] = value
    weight_dict = new_dict

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(
        num_classes,
        matcher,
        weight_dict,
        losses,
        focal_alpha=args.focal_alpha,
        reparam=args.reparam,
    )
    criterion.to(device)
    postprocessors = {
        "bbox": PostProcess(topk=args.topk, reparam=args.reparam)
    }
    if args.masks:
        postprocessors["segm"] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map,
                                                             threshold=0.85)

    return model, criterion, postprocessors


#$#>END: models/detr.py

#=============================================================================#
#                            models/transformer.py                            #
#=============================================================================#

#$#>START: models/transformer.py

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from util.misc import inverse_sigmoid
from util.box_ops import delta2bbox, box_xyxy_to_cxcywh
from models.utils import LayerNorm2D

from models.global_ape_decoder import build_global_ape_decoder
from models.global_rpe_decomp_decoder import build_global_rpe_decomp_decoder


class Transformer(nn.Module):

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_feature_levels=4,
        two_stage=False,
        two_stage_num_proposals=300,
        mixed_selection=False,
        norm_type='post_norm',
        decoder_type='deform',
        proposal_feature_levels=1,
        proposal_in_stride=16,
        proposal_tgt_strides=[8, 16, 32, 64],
        args=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        assert norm_type in ["pre_norm", "post_norm"], \
            f"expected norm type is pre_norm or post_norm, get {norm_type}"

        if decoder_type == 'global_ape':
            self.decoder = build_global_ape_decoder(args)
        elif decoder_type == 'global_rpe_decomp':
            self.decoder = build_global_rpe_decomp_decoder(args)
        else:
            raise NotImplementedError

        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self.mixed_selection = mixed_selection
        self.proposal_feature_levels = proposal_feature_levels
        self.proposal_tgt_strides = proposal_tgt_strides
        self.proposal_min_size = 50
        if two_stage and proposal_feature_levels > 1:
            assert len(proposal_tgt_strides) == proposal_feature_levels

            self.proposal_in_stride = proposal_in_stride
            self.enc_output_proj = nn.ModuleList([])
            for stride in proposal_tgt_strides:
                if stride == proposal_in_stride:
                    self.enc_output_proj.append(nn.Identity())
                elif stride > proposal_in_stride:
                    scale = int(math.log2(stride / proposal_in_stride))
                    layers = []
                    for _ in range(scale - 1):
                        layers += [
                            nn.Conv2d(d_model,
                                      d_model,
                                      kernel_size=2,
                                      stride=2),
                            LayerNorm2D(d_model),
                            nn.GELU()
                        ]
                    layers.append(
                        nn.Conv2d(d_model, d_model, kernel_size=2, stride=2))
                    self.enc_output_proj.append(nn.Sequential(*layers))
                else:
                    scale = int(math.log2(proposal_in_stride / stride))
                    layers = []
                    for _ in range(scale - 1):
                        layers += [
                            nn.ConvTranspose2d(d_model,
                                               d_model,
                                               kernel_size=2,
                                               stride=2),
                            LayerNorm2D(d_model),
                            nn.GELU()
                        ]
                    layers.append(
                        nn.ConvTranspose2d(d_model,
                                           d_model,
                                           kernel_size=2,
                                           stride=2))
                    self.enc_output_proj.append(nn.Sequential(*layers))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

        if hasattr(self.decoder, '_reset_parameters'):
            print('decoder re-init')
            self.decoder._reset_parameters()

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = self.d_model // 2
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats,
                             dtype=torch.float32,
                             device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes):
        if self.proposal_feature_levels > 1:
            memory, memory_padding_mask, spatial_shapes = self.expand_encoder_output(
                memory, memory_padding_mask, spatial_shapes)
        N_, S_, C_ = memory.shape
        # base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(
                N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0,
                               H_ - 1,
                               H_,
                               dtype=torch.float32,
                               device=memory.device),
                torch.linspace(0,
                               W_ - 1,
                               W_,
                               dtype=torch.float32,
                               device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(-1,
                                                                 keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        max_shape = None
        return output_memory, output_proposals, max_shape

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def expand_encoder_output(self, memory, memory_padding_mask,
                              spatial_shapes):
        assert spatial_shapes.size(
            0
        ) == 1, f'Get encoder output of shape {spatial_shapes}, not sure how to expand'

        bs, _, c = memory.shape
        h, w = spatial_shapes[0]

        _out_memory = memory.view(bs, h, w, c).permute(0, 3, 1, 2)
        _out_memory_padding_mask = memory_padding_mask.view(bs, h, w)

        out_memory, out_memory_padding_mask, out_spatial_shapes = [], [], []
        for i in range(self.proposal_feature_levels):
            mem = self.enc_output_proj[i](_out_memory)
            mask = F.interpolate(_out_memory_padding_mask[None].float(),
                                 size=mem.shape[-2:]).to(torch.bool)

            out_memory.append(mem)
            out_memory_padding_mask.append(mask.squeeze(0))
            out_spatial_shapes.append(mem.shape[-2:])

        out_memory = torch.cat(
            [mem.flatten(2).transpose(1, 2) for mem in out_memory], dim=1)
        out_memory_padding_mask = torch.cat(
            [mask.flatten(1) for mask in out_memory_padding_mask], dim=1)
        out_spatial_shapes = torch.as_tensor(out_spatial_shapes,
                                             dtype=torch.long,
                                             device=out_memory.device)
        return out_memory, out_memory_padding_mask, out_spatial_shapes

    def get_reference_points(self, memory, mask_flatten, spatial_shapes):
        output_memory, output_proposals, max_shape = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes)

        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](
            output_memory)
        enc_outputs_delta = None
        enc_outputs_coord_unact = (
            self.decoder.bbox_embed[self.decoder.num_layers](output_memory) +
            output_proposals)

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()
        return (reference_points, max_shape, enc_outputs_class,
                enc_outputs_coord_unact, enc_outputs_delta, output_proposals)

    def forward(self,
                srcs,
                masks,
                pos_embeds,
                query_embed=None,
                self_attn_mask=None):

        # TODO: we may remove this loop as we only have one feature level
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask,
                  pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes,
                                         dtype=torch.long,
                                         device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # prepare input for decoder
        memory = src_flatten
        bs, _, c = memory.shape
        if self.two_stage:
            (reference_points, max_shape, enc_outputs_class,
            enc_outputs_coord_unact, enc_outputs_delta, output_proposals) \
                = self.get_reference_points(memory, mask_flatten, spatial_shapes)
            init_reference_out = reference_points
            pos_trans_out = torch.zeros(
                (bs, self.two_stage_num_proposals, 2 * c),
                device=init_reference_out.device)
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(reference_points)))

            if not self.mixed_selection:
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            else:
                # query_embed here is the content embed for deformable DETR
                tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)
                query_embed, _ = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            lvl_pos_embed_flatten,
                                            spatial_shapes, level_start_index,
                                            valid_ratios, query_embed,
                                            mask_flatten, self_attn_mask,
                                            max_shape)

        inter_references_out = inter_references
        if self.two_stage:
            return (hs, init_reference_out, inter_references_out,
                    enc_outputs_class, enc_outputs_coord_unact,
                    enc_outputs_delta, output_proposals, max_shape)
        return hs, init_reference_out, inter_references_out, None, None, None, None, None


class TransformerReParam(Transformer):

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes):
        if self.proposal_feature_levels > 1:
            memory, memory_padding_mask, spatial_shapes = self.expand_encoder_output(
                memory, memory_padding_mask, spatial_shapes)
        N_, S_, C_ = memory.shape
        # base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            stride = self.proposal_tgt_strides[lvl]

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0,
                               H_ - 1,
                               H_,
                               dtype=torch.float32,
                               device=memory.device),
                torch.linspace(0,
                               W_ - 1,
                               W_,
                               dtype=torch.float32,
                               device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) * stride
            wh = torch.ones_like(grid) * self.proposal_min_size * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)

        H_, W_ = spatial_shapes[0]
        stride = self.proposal_tgt_strides[0]
        mask_flatten_ = memory_padding_mask[:, :H_ * W_].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1,
                            keepdim=True) * stride
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1,
                            keepdim=True) * stride
        img_size = torch.cat([valid_W, valid_H, valid_W, valid_H], dim=-1)
        img_size = img_size.unsqueeze(1)  # [BS, 1, 4]

        output_proposals_valid = ((output_proposals > 0.01 * img_size) &
                                  (output_proposals < 0.99 * img_size)).all(
                                      -1, keepdim=True)
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1).repeat(1, 1, 1),
            max(H_, W_) * stride,
        )
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid,
            max(H_, W_) * stride,
        )

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        max_shape = (valid_H[:, None, :], valid_W[:, None, :])
        return output_memory, output_proposals, max_shape

    def get_reference_points(self, memory, mask_flatten, spatial_shapes):
        output_memory, output_proposals, max_shape = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes)

        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](
            output_memory)
        enc_outputs_delta = self.decoder.bbox_embed[self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = box_xyxy_to_cxcywh(
            delta2bbox(output_proposals, enc_outputs_delta, max_shape))

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact
        return (reference_points, max_shape, enc_outputs_class,
                enc_outputs_coord_unact, enc_outputs_delta, output_proposals)


def build_transformer(args):
    model_class = Transformer if (not args.reparam) else TransformerReParam
    return model_class(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_feature_levels=args.num_feature_levels,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries_one2one +
        args.num_queries_one2many,
        mixed_selection=args.mixed_selection,
        norm_type=args.norm_type,
        decoder_type=args.decoder_type,
        proposal_feature_levels=args.proposal_feature_levels,
        proposal_in_stride=args.proposal_in_stride,
        proposal_tgt_strides=args.proposal_tgt_strides,
        args=args,
    )


#$#>END: models/transformer.py

#=============================================================================#
#                              models/matcher.py                              #
#=============================================================================#

#$#>START: models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, bbox2delta


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the
    predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because
    of this, in general, there are more predictions than targets. In this
    case, we do a 1-to-1 matching of the best predictions, while the others
    are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_bbox_type: str = "l1",
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification
            error in the matching cost cost_bbox: This is the relative weight
            of the L1 error of the bounding box coordinates in the matching
            cost cost_giou: This is the relative weight of the giou loss of
            the bounding box in the matching cost cost_bbox_type: This decides
            how to calculate box loss.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_bbox_type = cost_bbox_type
        assert (cost_class != 0 or cost_bbox != 0
                or cost_giou != 0), "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries,
                 num_classes] with the classification logits "pred_boxes":
                 Tensor of dim [batch_size, num_queries, 4] with the predicted
                 box coordinates

            targets: This is a list of targets (len(targets) = batch_size),
            where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where
                 num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the
                 target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j)
            where:
                - index_i is the indices of the selected predictions (in
                  order)
                - index_j is the indices of the corresponding selected targets
                  (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries,
                num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(
                0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = ((1 - alpha) * (out_prob**gamma) *
                              (-(1 - out_prob + 1e-8).log()))
            pos_cost_class = (alpha * ((1 - out_prob)**gamma) *
                              (-(out_prob + 1e-8).log()))
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:,
                                                                     tgt_ids]

            # Compute the L1 cost between boxes
            if self.cost_bbox_type == "l1":
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            elif self.cost_bbox_type == "reparam":
                out_delta = outputs["pred_deltas"].flatten(0, 1)
                out_bbox_old = outputs["pred_boxes_old"].flatten(0, 1)
                tgt_delta = bbox2delta(out_bbox_old, tgt_bbox)
                cost_bbox = torch.cdist(out_delta[:, None], tgt_delta,
                                        p=1).squeeze(1)
            else:
                raise NotImplementedError

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class +
                 self.cost_giou * cost_giou)
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [
                linear_sum_assignment(c[i])
                for i, c in enumerate(C.split(sizes, -1))
            ]
            return [(
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            ) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        cost_bbox_type='l1' if (not args.reparam) else 'reparam',
    )


#$#>END: models/matcher.py

#=============================================================================#
#                            models/segmentation.py                           #
#=============================================================================#

#$#>START: models/segmentation.py
"""
This file provides the definition of the convolutional heads used to predict masks,
 as well as the losses
"""
import io
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import util.box_ops as box_ops
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass


class DETRsegm(nn.Module):

    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim,
                                             hidden_dim,
                                             nheads,
                                             dropout=0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads,
                                           [1024, 512, 256], hidden_dim)

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.detr.backbone(samples)

        bs = features[-1].tensors.shape[0]

        src, mask = features[-1].decompose()
        src_proj = self.detr.input_proj(src)
        hs, memory = self.detr.transformer(src_proj, mask,
                                           self.detr.query_embed.weight,
                                           pos[-1])

        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1]
        }
        if self.detr.aux_loss:
            out["aux_outputs"] = [{
                "pred_logits": a,
                "pred_boxes": b
            } for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        # FIXME h_boxes takes the last one computed, keep this in mind
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)

        seg_masks = self.mask_head(
            src_proj,
            bbox_mask,
            [features[2].tensors, features[1].tensors, features[0].tensors],
        )
        outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries,
                                           seg_masks.shape[-2],
                                           seg_masks.shape[-1])

        out["pred_masks"] = outputs_seg_masks
        return out


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [
            dim,
            context_dim // 2,
            context_dim // 4,
            context_dim // 8,
            context_dim // 16,
            context_dim // 64,
        ]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_mask, fpns):

        def expand(tensor, length):
            return tensor.unsqueeze(1).repeat(1, int(length), 1, 1,
                                              1).flatten(0, 1)

        x = torch.cat([expand(x, bbox_mask.shape[1]),
                       bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax
    (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads)**-0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(k,
                     self.k_linear.weight.unsqueeze(-1).unsqueeze(-1),
                     self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads,
                    self.hidden_dim // self.num_heads)
        kh = k.view(
            k.shape[0],
            self.num_heads,
            self.hidden_dim // self.num_heads,
            k.shape[-2],
            k.shape[-1],
        )
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact,
                               kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs,
                       targets,
                       num_boxes,
                       alpha: float = 0.25,
                       gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                 targets,
                                                 reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcessSegm(nn.Module):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks,
                                      size=(max_h, max_w),
                                      mode="bilinear",
                                      align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(
                zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(results[i]["masks"].float(),
                                                size=tuple(tt.tolist()),
                                                mode="nearest").byte()

        return results


class PostProcessPanoptic(nn.Module):
    """This class converts the output of the model to the final panoptic
    result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the
           values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False)
                          class
           threshold: confidence threshold: segments with confidence lower
           than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's
        predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the
            model doc for the content. processed_sizes: This is a list of
            tuples (or torch tensors) of sizes of the images that were passed
            to the
                             model, ie the size after data augmentation but
                             before batching.
            target_sizes: This is a list of tuples (or torch tensors)
            corresponding to the requested final size
                          of each prediction. If left to None, it will default
                          to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = (
            outputs["pred_logits"],
            outputs["pred_masks"],
            outputs["pred_boxes"],
        )
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
                out_logits, raw_masks, raw_boxes, processed_sizes,
                target_sizes):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs["pred_logits"].shape[-1] -
                             1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[None],
                                    to_tuple(size),
                                    mode="bilinear").squeeze(0)
            cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each
            # stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w),
                                       dtype=torch.long,
                                       device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(
                    m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h),
                                         resample=Image.NEAREST)

                np_seg_img = (torch.ByteTensor(
                    torch.ByteStorage.from_buffer(seg_img.tobytes())).view(
                        final_h, final_w, 3).numpy())
                m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)],
                        dtype=torch.bool,
                        device=keep.device,
                    )
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(1,
                                         dtype=torch.long,
                                         device=cur_classes.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append({
                    "id": i,
                    "isthing": self.is_thing_map[cat],
                    "category_id": cat,
                    "area": a,
                })
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {
                    "png_string": out.getvalue(),
                    "segments_info": segments_info,
                }
            preds.append(predictions)
        return preds


#$#>END: models/segmentation.py

#=============================================================================#
#                                                                             #
#                    ██████  ██       █████  ██ ███    ██                     #
#                    ██   ██ ██      ██   ██ ██ ████   ██                     #
#                    ██████  ██      ███████ ██ ██ ██  ██                     #
#                    ██      ██      ██   ██ ██ ██  ██ ██                     #
#                    ██      ███████ ██   ██ ██ ██   ████                     #
#                                                                             #
#=============================================================================#
## STUFFS RELATED TO THEIR METHODS

#=============================================================================#
#                         models/global_ape_decoder.py                        #
#=============================================================================#

#$#>START: models/global_ape_decoder.py

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_

from util.misc import inverse_sigmoid, _get_clones, _get_activation_fn


class GlobalCrossAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query,
        k_input_flatten,
        v_input_flatten,
        input_padding_mask=None,
    ):

        B_, N, C = k_input_flatten.shape
        k = self.k(k_input_flatten).reshape(B_, N, self.num_heads,
                                            C // self.num_heads).permute(
                                                0, 2, 1, 3)
        v = self.v(v_input_flatten).reshape(B_, N, self.num_heads,
                                            C // self.num_heads).permute(
                                                0, 2, 1, 3)
        B_, N, C = query.shape
        q = self.q(query).reshape(B_, N, self.num_heads,
                                  C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        if input_padding_mask is not None:
            attn += input_padding_mask[:, None, None] * -100
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        norm_type='post_norm',
    ):
        super().__init__()

        self.norm_type = norm_type

        # global cross attention
        self.cross_attn = GlobalCrossAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model,
                                               n_heads,
                                               dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre(
        self,
        tgt,
        query_pos,
        src,
        src_pos_embed,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        # self attention
        tgt2 = self.norm2(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt2.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)

        # global cross attention
        tgt2 = self.norm1(tgt)
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt2, query_pos),
            self.with_pos_embed(src, src_pos_embed),
            src,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout4(tgt2)

        return tgt

    def forward_post(
        self,
        tgt,
        query_pos,
        src,
        src_pos_embed,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            self.with_pos_embed(src, src_pos_embed),
            src,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        src,
        src_pos_embed,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        if self.norm_type == "pre_norm":
            return self.forward_pre(tgt, query_pos, src, src_pos_embed,
                                    src_padding_mask, self_attn_mask)
        if self.norm_type == "post_norm":
            return self.forward_post(tgt, query_pos, src, src_pos_embed,
                                     src_padding_mask, self_attn_mask)


class GlobalDecoder(nn.Module):

    def __init__(
        self,
        decoder_layer,
        num_layers,
        return_intermediate=False,
        look_forward_twice=False,
        use_checkpoint=False,
        d_model=256,
        norm_type="post_norm",
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.look_forward_twice = look_forward_twice
        self.use_checkpoint = use_checkpoint
        # hack implementation for iterative bounding box
        # refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

        self.norm_type = norm_type
        if self.norm_type == "pre_norm":
            self.final_layer_norm = nn.LayerNorm(d_model)
        else:
            self.final_layer_norm = None

    def _reset_parameters(self):

        # stolen from Swin Transformer
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_pos_embed,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        self_attn_mask=None,
        max_shape=None,
    ):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if self.use_checkpoint:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    query_pos,
                    src,
                    src_pos_embed,
                    src_padding_mask,
                    self_attn_mask,
                )
            else:
                output = layer(
                    output,
                    query_pos,
                    src,
                    src_pos_embed,
                    src_padding_mask,
                    self_attn_mask,
                )

            if self.final_layer_norm is not None:
                output_after_norm = self.final_layer_norm(output)
            else:
                output_after_norm = output

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output_after_norm)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output_after_norm)
                intermediate_reference_points.append(
                    new_reference_points if self.
                    look_forward_twice else reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output_after_norm, reference_points


def build_global_ape_decoder(args):
    decoder_layer = GlobalDecoderLayer(
        d_model=args.hidden_dim,
        d_ffn=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        n_levels=args.num_feature_levels,
        n_heads=args.nheads,
        norm_type=args.norm_type,
    )
    decoder = GlobalDecoder(
        decoder_layer,
        num_layers=args.dec_layers,
        return_intermediate=True,
        look_forward_twice=args.look_forward_twice,
        use_checkpoint=args.decoder_use_checkpoint,
        d_model=args.hidden_dim,
        norm_type=args.norm_type,
    )
    return decoder


#$#>END: models/global_ape_decoder.py

#=============================================================================#
#                     models/global_rpe_decomp_decoder.py                     #
#=============================================================================#

#$#>START: models/global_rpe_decomp_decoder.py

# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import numpy as np
from timm.models.layers import trunc_normal_

from util.misc import inverse_sigmoid, _get_clones, _get_activation_fn
from util.box_ops import box_xyxy_to_cxcywh, delta2bbox


class GlobalCrossAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        rpe_hidden_dim=512,
        rpe_type='linear',
        feature_stride=16,
        reparam=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.rpe_type = rpe_type
        self.feature_stride = feature_stride
        self.reparam = reparam

        self.cpb_mlp1 = self.build_cpb_mlp(2, rpe_hidden_dim, num_heads)
        self.cpb_mlp2 = self.build_cpb_mlp(2, rpe_hidden_dim, num_heads)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
        cpb_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, out_dim, bias=False))
        return cpb_mlp

    def forward(
        self,
        query,
        reference_points,
        k_input_flatten,
        v_input_flatten,
        input_spatial_shapes,
        input_padding_mask=None,
    ):
        assert input_spatial_shapes.size(
            0) == 1, 'This is designed for single-scale decoder.'
        h, w = input_spatial_shapes[0]
        stride = self.feature_stride

        ref_pts = torch.cat([
            reference_points[:, :, :, :2] - reference_points[:, :, :, 2:] / 2,
            reference_points[:, :, :, :2] + reference_points[:, :, :, 2:] / 2,
        ],
                            dim=-1)  # B, nQ, 1, 4
        if not self.reparam:
            ref_pts[..., 0::2] *= (w * stride)
            ref_pts[..., 1::2] *= (h * stride)
        pos_x = torch.linspace(
            0.5, w - 0.5, w, dtype=torch.float32,
            device=w.device)[None, None, :, None] * stride  # 1, 1, w, 1
        pos_y = torch.linspace(
            0.5, h - 0.5, h, dtype=torch.float32,
            device=h.device)[None, None, :, None] * stride  # 1, 1, h, 1

        if self.rpe_type == 'abs_log8':
            delta_x = ref_pts[..., 0::2] - pos_x  # B, nQ, w, 2
            delta_y = ref_pts[..., 1::2] - pos_y  # B, nQ, h, 2
            delta_x = torch.sign(delta_x) * torch.log2(
                torch.abs(delta_x) + 1.0) / np.log2(8)
            delta_y = torch.sign(delta_y) * torch.log2(
                torch.abs(delta_y) + 1.0) / np.log2(8)
        elif self.rpe_type == 'linear':
            delta_x = ref_pts[..., 0::2] - pos_x  # B, nQ, w, 2
            delta_y = ref_pts[..., 1::2] - pos_y  # B, nQ, h, 2
        else:
            raise NotImplementedError

        rpe_x, rpe_y = self.cpb_mlp1(delta_x), self.cpb_mlp2(
            delta_y)  # B, nQ, w/h, nheads
        rpe = (rpe_x[:, :, None] + rpe_y[:, :, :, None]).flatten(
            2, 3)  # B, nQ, h, w, nheads ->  B, nQ, h*w, nheads
        rpe = rpe.permute(0, 3, 1, 2)

        B_, N, C = k_input_flatten.shape
        k = self.k(k_input_flatten).reshape(B_, N, self.num_heads,
                                            C // self.num_heads).permute(
                                                0, 2, 1, 3)
        v = self.v(v_input_flatten).reshape(B_, N, self.num_heads,
                                            C // self.num_heads).permute(
                                                0, 2, 1, 3)
        B_, N, C = query.shape
        q = self.q(query).reshape(B_, N, self.num_heads,
                                  C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn += rpe
        if input_padding_mask is not None:
            attn += input_padding_mask[:, None, None] * -100

        fmin, fmax = torch.finfo(attn.dtype).min, torch.finfo(attn.dtype).max
        torch.clip_(attn, min=fmin, max=fmax)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_heads=8,
        norm_type='post_norm',
        rpe_hidden_dim=512,
        rpe_type='box_norm',
        feature_stride=16,
        reparam=False,
    ):
        super().__init__()

        self.norm_type = norm_type

        # global cross attention
        self.cross_attn = GlobalCrossAttention(d_model,
                                               n_heads,
                                               rpe_hidden_dim=rpe_hidden_dim,
                                               rpe_type=rpe_type,
                                               feature_stride=feature_stride,
                                               reparam=reparam)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model,
                                               n_heads,
                                               dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_pos_embed,
        src_spatial_shapes,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        # self attention
        tgt2 = self.norm2(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt2.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)

        # global cross attention
        tgt2 = self.norm1(tgt)
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt2, query_pos),
            reference_points,
            self.with_pos_embed(src, src_pos_embed),
            src,
            src_spatial_shapes,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout4(tgt2)

        return tgt

    def forward_post(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_pos_embed,
        src_spatial_shapes,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            self.with_pos_embed(src, src_pos_embed),
            src,
            src_spatial_shapes,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_pos_embed,
        src_spatial_shapes,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        if self.norm_type == "pre_norm":
            return self.forward_pre(tgt, query_pos, reference_points, src,
                                    src_pos_embed, src_spatial_shapes,
                                    src_padding_mask, self_attn_mask)
        if self.norm_type == "post_norm":
            return self.forward_post(tgt, query_pos, reference_points, src,
                                     src_pos_embed, src_spatial_shapes,
                                     src_padding_mask, self_attn_mask)


class GlobalDecoder(nn.Module):

    def __init__(
        self,
        decoder_layer,
        num_layers,
        return_intermediate=False,
        look_forward_twice=False,
        use_checkpoint=False,
        d_model=256,
        norm_type="post_norm",
        reparam=False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.look_forward_twice = look_forward_twice
        self.use_checkpoint = use_checkpoint
        # hack implementation for iterative bounding box refinement and
        # two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.reparam = reparam

        self.norm_type = norm_type
        if self.norm_type == "pre_norm":
            self.final_layer_norm = nn.LayerNorm(d_model)
        else:
            self.final_layer_norm = None

    def _reset_parameters(self):

        # stolen from Swin Transformer
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_pos_embed,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        self_attn_mask=None,
        max_shape=None,
    ):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if self.reparam:
                reference_points_input = reference_points[:, :, None]
            else:
                if reference_points.shape[-1] == 4:
                    reference_points_input = (
                        reference_points[:, :, None] * torch.cat(
                            [src_valid_ratios, src_valid_ratios], -1)[:, None])
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = (reference_points[:, :, None] *
                                              src_valid_ratios[:, None])
            if self.use_checkpoint:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    query_pos,
                    reference_points_input,
                    src,
                    src_pos_embed,
                    src_spatial_shapes,
                    src_padding_mask,
                    self_attn_mask,
                )
            else:
                output = layer(
                    output,
                    query_pos,
                    reference_points_input,
                    src,
                    src_pos_embed,
                    src_spatial_shapes,
                    src_padding_mask,
                    self_attn_mask,
                )

            if self.final_layer_norm is not None:
                output_after_norm = self.final_layer_norm(output)
            else:
                output_after_norm = output

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output_after_norm)
                if reference_points.shape[-1] == 4:
                    if self.reparam:
                        new_reference_points = box_xyxy_to_cxcywh(
                            delta2bbox(reference_points, tmp, max_shape))
                    else:
                        new_reference_points = tmp + inverse_sigmoid(
                            reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                else:
                    if self.reparam:
                        raise NotImplementedError
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output_after_norm)
                intermediate_reference_points.append(
                    new_reference_points if self.
                    look_forward_twice else reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output_after_norm, reference_points


def build_global_rpe_decomp_decoder(args):
    decoder_layer = GlobalDecoderLayer(
        d_model=args.hidden_dim,
        d_ffn=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        n_heads=args.nheads,
        norm_type=args.norm_type,
        rpe_hidden_dim=args.decoder_rpe_hidden_dim,
        rpe_type=args.decoder_rpe_type,
        feature_stride=args.proposal_in_stride,
        reparam=args.reparam,
    )
    decoder = GlobalDecoder(
        decoder_layer,
        num_layers=args.dec_layers,
        return_intermediate=True,
        look_forward_twice=args.look_forward_twice,
        use_checkpoint=args.decoder_use_checkpoint,
        d_model=args.hidden_dim,
        norm_type=args.norm_type,
        reparam=args.reparam,
    )
    return decoder


#$#>END: models/global_rpe_decomp_decoder.py

#=============================================================================#
#                                                                             #
#     ██████   █████   ██████ ██   ██ ██████   ██████  ███    ██ ███████      #
#     ██   ██ ██   ██ ██      ██  ██  ██   ██ ██    ██ ████   ██ ██           #
#     ██████  ███████ ██      █████   ██████  ██    ██ ██ ██  ██ █████        #
#     ██   ██ ██   ██ ██      ██  ██  ██   ██ ██    ██ ██  ██ ██ ██           #
#     ██████  ██   ██  ██████ ██   ██ ██████   ██████  ██   ████ ███████      #
#                                                                             #
#=============================================================================#

#=============================================================================#
#                              models/backbone.py                             #
#=============================================================================#

#$#>START: models/backbone.py

import math
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .swin_transformer_v2 import SwinTransformerV2
from .utils import LayerNorm2D


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool,
                 return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (not train_backbone or "layer2" not in name
                    and "layer3" not in name and "layer4" not in name):
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2",
            # "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone,
                                            return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(),
                                 size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=norm_layer,
        )
        assert name not in ("resnet18",
                            "resnet34"), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class TransformerBackbone(nn.Module):

    def __init__(self, backbone: str, train_backbone: bool,
                 return_interm_layers: bool, args):
        super().__init__()
        out_indices = (1, 2, 3) if return_interm_layers else (3, )

        if backbone == "swin_v2_small_window16":
            backbone = SwinTransformerV2(
                pretrain_img_size=256,
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=16,
                drop_path_rate=args.drop_path_rate,
                use_checkpoint=args.use_checkpoint,
                out_indices=out_indices,
                pretrained_window_size=[16, 16, 16, 8],
                global_blocks=[[-1], [-1], [-1], [-1]])
            embed_dim = 96
            backbone.init_weights(args.pretrained_backbone_path)
        elif backbone == "swin_v2_small_window16_2global":
            backbone = SwinTransformerV2(
                pretrain_img_size=256,
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=16,
                drop_path_rate=args.drop_path_rate,
                use_checkpoint=args.use_checkpoint,
                out_indices=out_indices,
                pretrained_window_size=[16, 16, 16, 8],
                global_blocks=[[-1], [-1], [-1], [0, 1]])
            embed_dim = 96
            backbone.init_weights(args.pretrained_backbone_path)
        elif backbone == "swin_v2_small_window12to16":
            backbone = SwinTransformerV2(
                pretrain_img_size=256,
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=16,
                drop_path_rate=args.drop_path_rate,
                use_checkpoint=args.use_checkpoint,
                out_indices=out_indices,
                pretrained_window_size=[12, 12, 12, 6],
                global_blocks=[[-1], [-1], [-1], [-1]])
            embed_dim = 96
            backbone.init_weights(args.pretrained_backbone_path)
        elif backbone == "swin_v2_small_window12to16_2global":
            backbone = SwinTransformerV2(
                pretrain_img_size=256,
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=16,
                drop_path_rate=args.drop_path_rate,
                use_checkpoint=args.use_checkpoint,
                out_indices=out_indices,
                pretrained_window_size=[12, 12, 12, 6],
                global_blocks=[[-1], [-1], [-1], [0, 1]])
            embed_dim = 96
            backbone.init_weights(args.pretrained_backbone_path)
        else:
            raise NotImplementedError

        for name, parameter in backbone.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)

        if return_interm_layers:
            self.strides = [8, 16, 32]
            self.num_channels = [
                embed_dim * 2,
                embed_dim * 4,
                embed_dim * 8,
            ]
        else:
            self.strides = [32]
            self.num_channels = [embed_dim * 8]

        self.body = backbone

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(),
                                 size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class UpSampleWrapper(nn.Module):
    """Upsample last feat map to specific stride."""

    def __init__(
        self,
        net,
        stride,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            dim (int): number of channels of fpn hidden features.
        """
        super(UpSampleWrapper, self).__init__()

        self.net = net

        assert len(net.strides) == 1, 'UpSample should receive one input.'
        in_stride = net.strides[0]
        self.strides = [stride]

        assert len(net.num_channels) == 1, 'UpSample should receive one input.'
        in_num_channel = net.num_channels[0]

        assert stride <= in_stride, 'Target stride is larger than input stride.'
        if stride == in_stride:
            self.upsample = nn.Identity()
            self.num_channels = net.num_channels
        else:
            scale = int(math.log2(in_stride // stride))
            dim = in_num_channel
            layers = []
            for _ in range(scale - 1):
                layers += [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    LayerNorm2D(dim // 2),
                    nn.GELU()
                ]
                dim = dim // 2
            layers += [
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
            ]
            dim = dim // 2
            self.upsample = nn.Sequential(*layers)
            self.num_channels = [dim]

    def forward(self, tensor_list: NestedTensor):
        xs = self.net(tensor_list)

        assert len(xs) == 1

        out: Dict[str, NestedTensor] = {}
        for name, value in xs.items():
            m = tensor_list.mask
            assert m is not None
            x = self.upsample(value.tensors)
            mask = F.interpolate(m[None].float(),
                                 size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for idx, x in enumerate(out):
            pos.append(self[1][idx](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)

    if "resnet" in args.backbone:
        backbone = Backbone(
            args.backbone,
            train_backbone,
            return_interm_layers,
            args.dilation,
        )
    else:
        backbone = TransformerBackbone(args.backbone, train_backbone,
                                       return_interm_layers, args)

    if args.upsample_backbone_output:
        backbone = UpSampleWrapper(
            backbone,
            args.upsample_stride,
        )

    model = Joiner(backbone, position_embedding)
    return model


#$#>END: models/backbone.py

#=============================================================================#
#                        models/swin_transformer_v2.py                        #
#=============================================================================#

#$#>START: models/swin_transformer_v2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.utils.checkpoint as checkpoint

from torch import Tensor, Size
from typing import Union, List
from timm.models.layers import DropPath, to_2tuple, to_ntuple, trunc_normal_
import numpy as np

from .utils import load_swinv2_checkpoint

_shape_t = Union[int, List[int], Size]


def custom_normalize(input, p=2, dim=1, eps=1e-12, out=None):
    if out is None:
        denom = input.norm(p, dim, keepdim=True).expand_as(input)
        return input / (denom + eps)
    else:
        denom = input.norm(p, dim, keepdim=True).expand_as(input)
        return torch.div(input, denom + eps, out=out)


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
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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
    """ Window based multi-head self attention (W-MSA) module with relative
    position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels. window_size (tuple[int]): The
        height and width of the window. num_heads (int): Number of attention
        heads. qkv_bias (bool, optional):  If True, add a learnable bias to
        query, key, value. Default: True attn_drop (float, optional): Dropout
        ratio of attention weight. Default: 0.0 proj_drop (float, optional):
        Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(
            (num_heads, 1, 1))),
                                        requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     LinearFP32(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1),
                                         self.window_size[0],
                                         dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1),
                                         self.window_size[1],
                                         dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w])).permute(
                1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :,
                                  0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :,
                                  1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]
                                                            # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :,
                        0] += self.window_size[0] - 1  # shift to start from 0
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
        with torch.cuda.amp.autocast(enabled=False):
            qkv = F.linear(input=x.float(),
                           weight=self.qkv.weight,
                           bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = custom_normalize(q.float(), dim=-1, eps=5e-5)
        k = custom_normalize(k.float(), dim=-1, eps=5e-5)
        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(
                torch.tensor(1. / 0.01,
                             device=self.logit_scale.device))).exp()
        attn = (q @ k.transpose(-2, -1)) * logit_scale.float()

        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = attn.type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    #def extra_repr(self) -> str:
    #    return f'dim={self.dim}, window_size={self.window_size}, ' \
    #           f'pretrained_window_size={self.pretrained_window_size},
    #           num_heads={self.num_heads}'

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


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
        value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.
        Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, with L = {L}, H = {H}, W = {W}"

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
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

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
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class GlobalAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.pretrained_window_size = to_2tuple(
            pretrained_window_size)  # Wh, Ww
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(
            (num_heads, 1, 1))),
                                        requires_grad=True)
        # mlp to generate table of relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     LinearFP32(512, num_heads, bias=False))

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

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, H, W, C = x.shape
        N = H * W
        x = x.view(B, N, C)
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias,
                                               requires_grad=False),
                 self.v_bias))
        with torch.cuda.amp.autocast(enabled=False):
            qkv = F.linear(input=x.float(),
                           weight=self.qkv.weight,
                           bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = custom_normalize(q.float(), dim=-1, eps=5e-5)
        k = custom_normalize(k.float(), dim=-1, eps=5e-5)
        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(
                torch.tensor(1. / 0.01,
                             device=self.logit_scale.device))).exp()
        attn = (q @ k.transpose(-2, -1)) * logit_scale.float()


        relative_coords_h = torch.arange(-(H - 1), H,
                                         dtype=torch.float32).to(attn.device)
        relative_coords_w = torch.arange(-(W - 1), W,
                                         dtype=torch.float32).to(attn.device)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w
                            ])).permute(1, 2, 0).contiguous().unsqueeze(0)
        if self.pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :,
                                  0] /= (self.pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :,
                                  1] /= (self.pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (H - 1)
            relative_coords_table[:, :, :, 1] /= (W - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        coords_h = torch.arange(H).to(attn.device)
        coords_w = torch.arange(W).to(attn.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += H - 1  # shift to start from 0
        relative_coords[:, :, 1] += W - 1
        relative_coords[:, :, 0] *= 2 * W - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


        relative_position_bias_table = self.cpb_mlp(
            relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            relative_position_index.view(-1)].view(H * W, H * W,
                                                   -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = attn.type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    #def extra_repr(self) -> str:
    #    return f'dim={self.dim},'
    #    pretrained_window_size={self.pretrained_window_size}
    #    num_heads={self.num_heads}'


class SwinTransformerGlobalBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query,
        key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = GlobalAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):

        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, with L = {L}, H = {H}, W = {W}"

        # ATTN
        shortcut = x
        x = x.view(B, H, W, C)
        x = self.attn(x)
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

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

        x = self.reduction(x)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels depth (int): Number of blocks.
        num_heads (int): Number of attention heads. window_size (int): Local
        window size. mlp_ratio (float): Ratio of mlp hidden dim to embedding
        dim. qkv_bias (bool, optional): If True, add a learnable bias to
        query, key, value. Default: True drop (float, optional): Dropout rate.
        Default: 0.0 attn_drop (float, optional): Attention dropout rate.
        Default: 0.0 drop_path (float | tuple[float], optional): Stochastic
        depth rate. Default: 0.0 norm_layer (nn.Module, optional):
        Normalization layer. Default: nn.LayerNorm downsample (nn.Module |
        None, optional): Downsample layer at the end of the layer. Default:
        None use_checkpoint (bool): Whether to use checkpointing to save
        memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pretrained_window_size=0,
                 global_blocks=[-1]):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if
                                 (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            if i not in global_blocks else SwinTransformerGlobalBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size)
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
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

        mask_windows = window_partition(
            img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                          float(-100.0)).masked_fill(
                                              attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            if isinstance(self.downsample, PatchMerging):
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
            else:
                Wh, Ww = H, W
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

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


class SwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer
         using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained
        model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4 in_chans (int):
        Number of input image channels. Default: 3 embed_dim (int): Patch
        embedding dimension. Default: 96 depths (tuple(int)): Depth of each
        Swin Transformer layer. num_heads (tuple(int)): Number of attention
        heads in different layers. window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default:
        4 qkv_bias (bool): If True, add a learnable bias to query, key, value.
        Default: True drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1 norm_layer
        (nn.Module): Normalization layer. Default: nn.LayerNorm. ape (bool):
        If True, add absolute position embedding to the patch embedding.
        Default: False patch_norm (bool): If True, add normalization after
        patch embedding. Default: True use_checkpoint (bool|tuple(bool)):
        Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each
        layer. out_indices (Sequence(int)): Output from which stages. Default:
        (0, 1, 2, 3) global_blocks (Sequence(Sequence(int))): Global attention
        blocks in each stage. frozen_stages (int): Stages to be frozen (stop
        grad and set eval mode).
            -1 means not freezing any parameters.
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
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=partial(LayerNormFP32, eps=1e-6),
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 pretrained_window_size=[0, 0, 0, 0],
                 out_indices=(0, 1, 2, 3),
                 global_blocks=[[-1], [-1], [-1], [-1]],
                 frozen_stages=-1):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.depths = depths
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.mlp_ratio = mlp_ratio

        use_checkpoint = to_ntuple(self.num_layers)(use_checkpoint)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
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

        # build layers
        self.layers = nn.ModuleList()
        num_features = []
        for i_layer in range(self.num_layers):
            cur_dim = int(embed_dim * 2**i_layer)
            num_features.append(cur_dim)
            layer = BasicLayer(
                dim=cur_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if
                (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint[i_layer],
                pretrained_window_size=pretrained_window_size[i_layer],
                global_blocks=global_blocks[i_layer],
            )
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        if isinstance(pretrained, str):
            load_swinv2_checkpoint(self,
                                   pretrained,
                                   strict=False,
                                   map_location='cpu')
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

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

        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = (x_out.view(-1, H, W, self.num_features[i]).permute(
                    0, 3, 1, 2).contiguous())
                outs[str(i)] = out

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformerV2, self).train(mode)
        self._freeze_stages()


#$#>END: models/swin_transformer_v2.py

#=============================================================================#
#                         models/position_encoding.py                         #
#=============================================================================#

#$#>START: models/position_encoding.py
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale
        else:
            y_embed = (y_embed - 0.5) * self.scale
            x_embed = (x_embed - 0.5) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32,
                             device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (torch.cat(
            [
                x_emb.unsqueeze(0).repeat(h, 1, 1),
                y_emb.unsqueeze(1).repeat(1, w, 1),
            ],
            dim=-1,
        ).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1))
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)
    elif args.position_embedding in ("v4", "sine_unnorm"):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=False)
    else:
        raise ValueError(f"not supported {args.position_embedding}")
    position_embedding = nn.ModuleList(
        [position_embedding for _ in range(args.num_feature_levels)])

    return position_embedding


#$#>END: models/position_encoding.py

#=============================================================================#
#                                                                             #
#                    ██    ██ ████████ ██ ██      ███████                     #
#                    ██    ██    ██    ██ ██      ██                          #
#                    ██    ██    ██    ██ ██      ███████                     #
#                    ██    ██    ██    ██ ██           ██                     #
#                     ██████     ██    ██ ███████ ███████                     #
#                                                                             #
#=============================================================================#

#=============================================================================#
#                               models/utils.py                               #
#=============================================================================#

#$#>START: models/utils.py

import torch
import torch.nn as nn


class Sample2D(nn.Module):

    def __init__(self, stride, start):
        super().__init__()
        self.stride = stride
        self.start = start

    def forward(self, x):
        """
        x: N C H W
        """
        _, _, h, w = x.shape
        return x[:, :, self.start:h:self.stride, self.start:w:self.stride]

    def extra_repr(self) -> str:
        return f'stride={self.stride}, start={self.start}'


class LayerNorm2D(nn.Module):

    def __init__(self, normalized_shape, norm_layer=nn.LayerNorm):
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


def load_swinv2_checkpoint(model, filename, map_location="cpu", strict=False):
    """Load swin checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    if filename.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            filename, map_location=map_location, check_hash=True)
    else:
        checkpoint = torch.load(filename, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"No state_dict found in checkpoint file {filename}")
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # strip prefix for mim ckpt
    if any([k.startswith("encoder.") for k in state_dict.keys()]):
        print("Remove encoder. prefix")
        state_dict = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items() if k.startswith("encoder.")
        }

    # rename rpe to cpb (naming inconsistency of sup & mim ckpt)
    if any(["rpe_mlp" in k for k in state_dict.keys()]):
        print("Replace rpe_mlp with cpb_mlp")
        state_dict = {
            k.replace("rpe_mlp", "cpb_mlp"): v
            for k, v in state_dict.items()
        }

    # remove relative_coords_table & relative_position_index in state_dict as
    # they would be re-init
    if any([
            "relative_coords_table" in k or "relative_position_index" in k
            for k in state_dict.keys()
    ]):
        print(
            "Remove relative_coords_table & relative_position_index (they would be re-init)"
        )
        state_dict = {
            k: v
            for k, v in state_dict.items() if "relative_coords_table" not in k
            and "relative_position_index" not in k
        }

    # reshape absolute position embedding
    if state_dict.get("absolute_pos_embed") is not None:
        absolute_pos_embed = state_dict["absolute_pos_embed"]
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            print("Warning: Error in loading absolute_pos_embed, pass")
        else:
            state_dict["absolute_pos_embed"] = absolute_pos_embed.view(
                N2, H, W, C2).permute(0, 3, 1, 2)

    # load state_dict
    msg = model.load_state_dict(state_dict, strict=strict)
    print(msg)
    return checkpoint


#$#>END: models/utils.py
