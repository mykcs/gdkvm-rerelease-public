import logging
import math
from typing import Tuple, Iterable, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def gdr_decay(A_log: torch.Tensor, projected_alpha: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """Convert the learned log-space GDR gate into a stable multiplicative decay.

    The projected gate is parameterized in log space as
    g = -exp(A_log) * softplus(projected_alpha + dt_bias), so g <= 0.
    The recurrent state must be multiplied by exp(g), yielding a decay in (0, 1].
    """
    log_decay = -A_log.float().exp() * F.softplus(projected_alpha.float() + dt_bias)
    return torch.exp(log_decay)

from model.utils import resnet
from model.aux_modules import AuxComputer
from model.transformer.object_transformer import QueryTransformer
from model.transformer.object_summarizer import ObjectSummarizer
from utils.tensor_utils import aggregate

log = logging.getLogger()

from model.modules.modules import ImageEncoder, MaskEncoder, MaskDecoder
from model.kpff import KPFF, KeyProj, PixProj, PixelFuser, MultiscaleSensoryUpdater, SensoryUpdater, FeatureFuser


class GDKVM(nn.Module):
    def __init__(
            self,
            model_type='base',
            image_encoder_type='resnet50',
            mask_encoder_type='resnet18',
            use_channels_last: bool=False,
            pretrained_backbones: bool=True,
    ) -> None:
        super().__init__()
        self.use_channels_last = bool(use_channels_last)
        self.pretrained_backbones = bool(pretrained_backbones)
        self.ms_dims = {
            'resnet50': [1024, 512, 256], 
            'resnet18': [256, 128, 64],
        }[image_encoder_type]

        self.up_dims = {
            'base': [256, 128, 128],
            'small': [256, 128, 64]
        }[model_type]

        self.key_dim = 64
        self.value_dim = 256
        self.pixel_dim = 256
        self.sensory_dim = 256
        self.embed_dim = 256

        # Parameters for the query transformer
        self.num_blocks = 3
        self.num_heads = 8
        self.num_queries = 16
        self.ff_dim = 2048

        self.image_encoder = ImageEncoder(
            encoder_type=image_encoder_type,
            pretrained=self.pretrained_backbones,
        )
        self.mask_encoder = MaskEncoder(
            self.pixel_dim,
            self.value_dim,
            self.sensory_dim,
            encoder_type=mask_encoder_type,
            pretrained=self.pretrained_backbones,
        )
        
        self.key_projector = KeyProj(self.ms_dims[0], self.pixel_dim, self.key_dim)
        self.pix_projector = PixProj(self.ms_dims[0], self.pixel_dim)
        self.KPFF          = KPFF   (self.ms_dims[0], self.pixel_dim, self.key_dim)
        self.mask_decoder = MaskDecoder(self.ms_dims, self.up_dims, self.sensory_dim)
        self.pixel_fuser = PixelFuser(
            self.pixel_dim, self.embed_dim, self.value_dim, self.sensory_dim)

        self.object_transformer = QueryTransformer(
            self.value_dim, self.embed_dim, self.ff_dim, self.num_blocks,
            self.num_heads, self.num_queries)

        self.object_summarizer = ObjectSummarizer(
            self.value_dim, self.embed_dim, self.num_queries, add_pe=True)

        self.aux_computer = AuxComputer(self.sensory_dim, self.embed_dim)

        self.register_buffer(
            "pixel_mean", torch.Tensor([0.5]).view(-1, 1, 1), False)
        self.register_buffer(
            "pixel_std", torch.Tensor([0.5]).view(-1, 1, 1), False)
        
        self.b_proj = nn.Linear(self.value_dim, self.num_heads, bias=False)
        self.a_proj = nn.Linear(self.value_dim, self.num_heads, bias=False)
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True
        

    def encode_image(
        self, image: torch.Tensor
    ) -> Tuple[Iterable[torch.Tensor], torch.Tensor]:
        
        image = (image - self.pixel_mean) / self.pixel_std
        multiscale_feats = self.image_encoder(image)
        return multiscale_feats

    def encode_mask(
        self, 
        image_BCHW: torch.Tensor, 
        pixfeat_BCHW: torch.Tensor, 
        mask_BNHW: torch.Tensor,
        sensory_BNCHW: torch.Tensor,
        *,
        deep_update: bool=False,
        chunk_size: int=-1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image (tensor): B, C, H, W.
            pixfeat (tensor): B, C, H, W.
            mask (tensor): B, N, H, W. No background in the mask.
        Returns:
            value (tensor): shape of BNCHW.
        """
        image_BCHW = (image_BCHW - self.pixel_mean) / self.pixel_std
        mask_other_BNHW = (mask_BNHW.float().sum(dim=1, keepdim=True) - mask_BNHW.float()).clamp(0, 1)
        masks_BN2HW = torch.stack([mask_BNHW, mask_other_BNHW], dim=2)
        mask_value_BNCHW, sensory_BNCHW = self.mask_encoder(
            image_BCHW, pixfeat_BCHW, masks_BN2HW, sensory_BNCHW, 
            deep_update=deep_update, chunk_size=chunk_size)

        object_memory_BNQC = self.object_summarizer(mask_BNHW, mask_value_BNCHW)

        return mask_value_BNCHW, sensory_BNCHW, object_memory_BNQC

    def pixel_fusion(
        self, 
        pixfeat_BCHW: torch.Tensor, 
        readout_BNCHW: torch.Tensor, 
        sensory_BNCHW: torch.Tensor,
        last_masks_BNHW: torch.Tensor,
    ) -> torch.Tensor:
        masks_BNHW = F.interpolate(
            last_masks_BNHW.float(), size=readout_BNCHW.shape[-2:], mode='area')
        masks_sum_B1HW = masks_BNHW.sum(dim=1, keepdim=True)
        masks_others_BNHW = (masks_sum_B1HW - masks_BNHW).clamp(0, 1)

        readout_fused_BNCHW = self.pixel_fuser(
            pixfeat_BCHW, readout_BNCHW, sensory_BNCHW, masks_BNHW, masks_others_BNHW)
        return readout_fused_BNCHW

    def segment(
        self, 
        ms_feats: Tuple[torch.Tensor], 
        readout_BNCHW: torch.Tensor, 
        pixfeat_BCHW: torch.Tensor,
        last_masks_BNHW: torch.Tensor,
        sensory_BNCHW: torch.Tensor,
        object_memory_BNQC: torch.Tensor,
        *,
        selector: Optional[torch.Tensor] = None,
        update_sensory: bool=False,
        chunk_size: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            ms_feats: multi-scale features extracted from image
            value (tensor): mask value
        """
        readout_BNCHW = self.pixel_fusion(
            pixfeat_BCHW, readout_BNCHW, sensory_BNCHW, last_masks_BNHW)

        # Enhance the readout with object memory
        readout_BNCHW, aux_features = self.object_transformer(
            readout_BNCHW, object_memory_BNQC, selector=selector)

        logits, sensory_BNCHW = self.mask_decoder(
            ms_feats, readout_BNCHW, sensory_BNCHW,
            chunk_size=chunk_size, update_sensory=update_sensory)

        masks = torch.sigmoid(logits)
        if selector is not None:
            masks = masks * selector

        # Softmax over all objects
        logits = aggregate(masks, dim=1)
        logits = F.interpolate(
            logits, scale_factor=4, mode='bilinear', align_corners=False)
        masks = F.softmax(logits, dim=1)

        return logits, masks, sensory_BNCHW, aux_features

    def forward(self, data: Dict):
        out                = {}
        raw_num_objects = data['info']['num_objects']
        if isinstance(raw_num_objects, torch.Tensor):
            raise TypeError(
                "info.num_objects must be normalized to Python integers before GDKVM.forward"
            )
        if isinstance(raw_num_objects, (int, float)):
            raw_num_objects = [raw_num_objects]
        num_objects = [int(num) for num in raw_num_objects]
        out['num_objects'] = num_objects
        
        max_num_objects = max(num_objects)

        images_BTCHW = data['rgb']
        B, T         = images_BTCHW.shape[:2]
        
        # Optimization: Use channels_last memory format for acceleration.
        images_DCHW  = images_BTCHW.reshape(B*T, *images_BTCHW.shape[2:]) 
        if self.use_channels_last and images_DCHW.device.type == 'cuda':
            images_DCHW = images_DCHW.to(memory_format=torch.channels_last)
        
        ms_feats      = self.encode_image(images_DCHW)

        # Optimization: Reshape features outside the loop.
        ms_feats_reshaped = [f.view(B, T, *f.shape[1:]) for f in ms_feats]

        feat_DCHW     = ms_feats[0]
        pixfeat_DCHW  = self.pix_projector(feat_DCHW)
        pixfeat_BTCHW = pixfeat_DCHW.view(B, T, *pixfeat_DCHW.shape[1:])

        # 其他投影等
        key_DCHW  = self.key_projector(feat_DCHW)
        key_DCHW = self.KPFF(key_DCHW, pixfeat_DCHW)
        key_BTCHW = key_DCHW.view(B, T, *key_DCHW.shape[1:])

        # Extract 0-th frame image and mask.
        first_frame_image_BCHW    = images_BTCHW[:, 0]
        first_frame_pixfeat_BCHW  = pixfeat_BTCHW[:, 0]

        first_frame_mask_B1MHW = data['ff_gt']
        first_frame_mask_B1MHW = torch.zeros_like(first_frame_mask_B1MHW)
        first_frame_mask_BNHW  = first_frame_mask_B1MHW[:, 0, :max_num_objects]

        # Initialize sensory.
        N, Cs = max_num_objects, self.sensory_dim
        Hk, Wk = key_DCHW.shape[2:]
        sensory_BNCHW = torch.zeros(B, N, Cs, Hk, Wk, device=key_DCHW.device)
        
        # Encode mask for the 0-th frame.
        value_BNCHW, sensory_BNCHW, object_memory_BNQC = self.encode_mask(
            first_frame_image_BCHW, first_frame_pixfeat_BCHW,
            first_frame_mask_BNHW, sensory_BNCHW, deep_update=True
        )

        # Obtain initial object memory.
        object_memory_sum_BNQC = object_memory_BNQC.clone()

        # Compute initial state.
        key_BCHW = key_BTCHW[:, 0]
        key_max_B1HW = torch.max(key_BCHW, dim=1, keepdim=True).values
        key_BCHW = (key_BCHW - key_max_B1HW).softmax(dim=1)
        state_BNCC = torch.einsum('bkhw,bnvhw->bnkv', key_BCHW, value_BNCHW)

        last_masks_BNHW = first_frame_mask_BNHW.clone()

        # t0
        this_key_BCHW = key_BTCHW[:, 0]
        this_key_max_B1HW = torch.max(this_key_BCHW, dim=1, keepdim=True).values
        this_key_BCHW = (this_key_BCHW - this_key_max_B1HW).softmax(dim=1)
        this_readout_BNCHW = torch.einsum(
            'bkhw,bnkv->bnvhw', this_key_BCHW, state_BNCC).contiguous()

        this_pixfeat_BCHW = pixfeat_BTCHW[:, 0]
        # Optimization: Slice pre-processed features.
        this_ms_feats = [f[:, 0] for f in ms_feats_reshaped]
        this_logits_BNHW, this_masks_BNHW, sensory_BNCHW, aux = self.segment(
                this_ms_feats, 
                this_readout_BNCHW, 
                this_pixfeat_BCHW, 
                last_masks_BNHW, 
                sensory_BNCHW, 
                object_memory_sum_BNQC, 
                update_sensory=True)
        this_aux_output = self.aux_computer(this_pixfeat_BCHW, sensory_BNCHW, aux)
        last_masks_BNHW = this_masks_BNHW[:, 1:]
        out[f'logits_{0}'] = this_logits_BNHW
        out[f'masks_{0}'] = this_masks_BNHW[:, 1:]  # remove the background
        out[f'aux_{0}'] = this_aux_output

        for i in range(1, T):
            # t-1
            state_t_1_BNCC = state_BNCC.clone()

            # Eraser term
            this_key_BCHW = key_BTCHW[:, i]
            this_key_max_B1HW = torch.max(this_key_BCHW, dim=1, keepdim=True).values
            this_key_BCHW = (this_key_BCHW - this_key_max_B1HW).softmax(dim=1)
            
            v_old = torch.einsum('bkhw,bnkv->bnvhw', this_key_BCHW, state_t_1_BNCC).contiguous()
            v_k = torch.einsum('bkhw,bnvhw->bnkv', this_key_BCHW, v_old)
            
            beta_t = self.b_proj(state_t_1_BNCC).sigmoid()
            beta_t_expan_b = beta_t.repeat_interleave(
                self.value_dim // self.num_heads, dim=3
            )
            eraser = torch.einsum('bnkv,bnkv->bnkv', beta_t_expan_b, v_k).contiguous()

            # New information
            this_image_BCHW = images_BTCHW[:, i]
            this_value_BNCHW, sensory_BNCHW, object_memory_BNQC = self.encode_mask(
                this_image_BCHW, this_pixfeat_BCHW,
                last_masks_BNHW,
                sensory_BNCHW, deep_update=True)
            
            vk_t = torch.einsum('bkhw,bnvhw->bnkv', this_key_BCHW, this_value_BNCHW).contiguous()
            new = torch.einsum('bnkv,bnkv->bnkv', beta_t_expan_b, vk_t).contiguous()

            # Alpha
            old = state_t_1_BNCC - eraser
            alpha = gdr_decay(
                self.A_log,
                self.a_proj(state_t_1_BNCC),
                self.dt_bias,
            )

            # Expand head-wise decay over the value channels.
            alpha_expanded = alpha.repeat_interleave(
                self.value_dim // self.num_heads,
                dim=3,
            )
            old = torch.einsum('bnkv,bnkv->bnkv', alpha_expanded, old).contiguous()

            # End
            state_BNCC = old + new

            # Readout
            this_readout_BNCHW = torch.einsum(
                'bkhw,bnkv->bnvhw', this_key_BCHW, state_BNCC).contiguous()

            # Segment
            this_pixfeat_BCHW = pixfeat_BTCHW[:, i]
            # Optimization: Slice directly.
            this_ms_feats = [f[:, i] for f in ms_feats_reshaped]
            
            this_logits_BNHW, this_masks_BNHW, sensory_BNCHW, aux = self.segment(
                this_ms_feats, 
                this_readout_BNCHW, 
                this_pixfeat_BCHW, 
                last_masks_BNHW, 
                sensory_BNCHW, 
                object_memory_sum_BNQC, 
                update_sensory=True)
            
            # Auxilary task
            this_aux_output = self.aux_computer(this_pixfeat_BCHW, sensory_BNCHW, aux)

            last_masks_BNHW = this_masks_BNHW[:, 1:]  # remove the background

            # Accumulate object memory.
            object_memory_sum_BNQC = object_memory_sum_BNQC + object_memory_BNQC

            out[f'logits_{i}'] = this_logits_BNHW
            out[f'masks_{i}'] = this_masks_BNHW[:, 1:]  # remove the background
            out[f'aux_{i}'] = this_aux_output
        
        return out
    def load_weights(self, src_dict, init_as_zero_if_needed=False) -> None:
        # Map single-object weight to multi-object weight (4->5 out channels in conv1)
        for k in list(src_dict.keys()):
            if k == 'mask_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    log.info(f'Converting {k} from single object to multiple objects.')
                    pads = torch.zeros((64, 1, 7, 7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        nn.init.orthogonal_(pads)
                        log.info(f'Randomly initialized padding for {k}.')
                    else:
                        log.info(f'Zero-initialized padding for {k}.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)
            elif k == 'pixel_fuser.sensory_compress.weight':
                if src_dict[k].shape[1] == self.sensory_dim + 1:
                    log.info(f'Converting {k} from single object to multiple objects.')
                    pads = torch.zeros((self.value_dim, 1, 1, 1), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        nn.init.orthogonal_(pads)
                        log.info(f'Randomly initialized padding for {k}.')
                    else:
                        log.info(f'Zero-initialized padding for {k}.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        for k in src_dict:
            if k not in self.state_dict():
                log.info(f'Key {k} found in src_dict but not in self.state_dict()!!!')
        for k in self.state_dict():
            if k not in src_dict:
                log.info(f'Key {k} found in self.state_dict() but not in src_dict!!!')

        self.load_state_dict(src_dict, strict=False)
