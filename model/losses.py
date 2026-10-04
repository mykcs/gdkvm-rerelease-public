from typing import List, Dict, Tuple
from omegaconf import DictConfig
from collections import defaultdict
import torch
import torch.nn.functional as F

# 假设这些工具函数依然从 utils 导入
from utils.point_features import calculate_uncertainty, point_sample, get_uncertain_point_coords_with_randomness
from utils.tensor_utils import cls_to_one_hot

@torch.jit.script
def ce_loss_jit(log_probs: torch.Tensor, soft_gt: torch.Tensor) -> torch.Tensor:
    """
    计算 Cross Entropy Loss。
    优化：接收 log_probs 而不是 logits，避免重复计算 log_softmax。
    Args:
        log_probs: T x C x N (已经经过 LogSoftmax)
        soft_gt: T x C x N (One-hot 或 Soft Label)
    """
    # 手动计算 Cross Entropy: - sum(target * log_prob)
    # dim=1 是 Channel 维度
    loss = -(soft_gt * log_probs).sum(dim=1)
    # sum over temporal dimension (dim 0), then mean over points
    return loss.sum(0).mean()

@torch.jit.script
def dice_loss_jit(probs: torch.Tensor, soft_gt: torch.Tensor) -> torch.Tensor:
    """
    计算 Dice Loss。
    优化：移除 flatten，直接在最后两维进行 reduce，节省显存。
    Args:
        probs: T x C x N (已经经过 Softmax/Exp)
        soft_gt: T x C x N
    """
    # 忽略背景类 (C=0)
    # 输入形状: [T, C, N]
    pred = probs[:, 1:]
    gt = soft_gt[:, 1:]

    # 直接在 Channel 和 Points 维度上计算，或者仅在 Points 维度计算
    # 原版逻辑是 flatten 后对所有点求和。这里等价于对最后一维(N)求和。
    
    intersection = (pred * gt).sum(dim=-1)
    union = pred.sum(dim=-1) + gt.sum(dim=-1)
    
    # Dice score: 2 * inter / (union + epsilon)
    # Loss: 1 - dice
    loss = 1 - (2 * intersection + 1) / (union + 1)
    
    # sum over temporal, mean over others (if any remaining)
    return loss.sum(0).mean()



def _stack_per_timestep(
    data: Dict[str, torch.Tensor],
    key_prefix: str,
    batch_idx: int,
    t_range,
    valid_slice,
    is_aux: bool = False,
    aux_list=None,
    level_idx: int = -1,
) -> torch.Tensor:
    """B3: Module-level helper extracted from compute() to avoid 4-level closure."""
    tensors = []
    for i, ti in enumerate(t_range):
        if is_aux:
            src = aux_list[i][key_prefix]
        else:
            src = data[f'{key_prefix}_{ti}']
        if level_idx >= 0:
            tensors.append(src[batch_idx, valid_slice, level_idx])
        else:
            tensors.append(src[batch_idx, valid_slice])
    return torch.stack(tensors, dim=0)


class LossComputer:
    def __init__(self, cfg: DictConfig, stage_cfg: DictConfig):
        # super().__init__() # Not needed for object base
        self.point_supervision = stage_cfg.point_supervision
        self.num_points = stage_cfg.train_num_points
        self.oversample_ratio = stage_cfg.oversample_ratio
        self.importance_sample_ratio = stage_cfg.importance_sample_ratio

        self.sensory_weight = cfg.model.aux_loss.sensory.weight
        self.query_weight = cfg.model.aux_loss.query.weight

    def mask_loss(
        self, logits: torch.Tensor, soft_gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算点监督的 CE 和 Dice Loss。
        logits/soft_gt shape: [T, C, H, W]
        """
        assert self.point_supervision

        with torch.no_grad():
            # 1. 采样不确定点
            # logits shape: [T, C, H, W] -> point_coords: [T, NumPoints, 2]
            point_coords = get_uncertain_point_coords_with_randomness(
                logits, lambda x: calculate_uncertainty(x), 
                self.num_points, self.oversample_ratio, self.importance_sample_ratio
            )
            # 2. 获取对应点的 GT
            point_labels = point_sample(soft_gt, point_coords, align_corners=False)
        
        # 3. 获取对应点的预测值
        point_logits = point_sample(logits, point_coords, align_corners=False)
        # point_labels / point_logits shape: [T, C, num_points]

        # --- 核心优化部分 ---
        # 统一计算 LogSoftmax，避免在 CE 和 Dice 中重复进行指数运算
        point_log_probs = F.log_softmax(point_logits, dim=1)
        point_probs = point_log_probs.exp() # 用于 Dice

        # 调用 JIT 编译的 Loss 函数
        loss_ce = ce_loss_jit(point_log_probs, point_labels)
        loss_dice = dice_loss_jit(point_probs, point_labels)

        return loss_ce, loss_dice

    def compute(self, data: Dict[str, torch.Tensor],
                num_objects: List[int]) -> Dict[str, torch.Tensor]:
        batch_size, num_frames = data['rgb'].shape[:2]
        losses = defaultdict(float)

        # 只有第一帧和最后一帧参与计算（如果序列长度 > 1）
        t_range = [0, num_frames - 1] if num_frames > 1 else [0]

        # 预先提取 Aux 数据列表，避免在循环内部重复查找字典
        aux_list = [data[f'aux_{ti}'] for ti in t_range]
        has_sensory = 'sensory_logits' in aux_list[0]
        has_query = 'q_logits' in aux_list[0]

        for bi in range(batch_size):
            curr_obj_count = num_objects[bi]
            valid_slice = slice(None, curr_obj_count + 1)

            # 1. 准备 Ground Truth
            cls_gt = data['cls_gt'][bi, t_range]
            soft_gt = cls_to_one_hot(cls_gt, curr_obj_count)

            # 2. Main Loss
            logits = _stack_per_timestep(
                data, 'logits', bi, t_range, valid_slice)
            l_ce, l_dice = self.mask_loss(logits, soft_gt)
            losses['loss_ce'] += l_ce / batch_size
            losses['loss_dice'] += l_dice / batch_size

            # 3. Auxiliary Losses - Sensory
            if has_sensory:
                sensory_log = _stack_per_timestep(
                    data, 'sensory_logits', bi, t_range, valid_slice, is_aux=True, aux_list=aux_list)
                l_ce, l_dice = self.mask_loss(sensory_log, soft_gt)
                losses['aux_sensory_ce'] += l_ce / batch_size * self.sensory_weight
                losses['aux_sensory_dice'] += l_dice / batch_size * self.sensory_weight

            # 4. Auxiliary Losses - Query
            if has_query:
                num_levels = aux_list[0]['q_logits'].shape[2]
                for level_idx in range(num_levels):
                    query_log = _stack_per_timestep(
                        data, 'q_logits', bi, t_range, valid_slice,
                        is_aux=True, aux_list=aux_list, level_idx=level_idx)
                    l_ce, l_dice = self.mask_loss(query_log, soft_gt)
                    w = self.query_weight / batch_size
                    losses[f'aux_query_ce_l{level_idx}'] += l_ce * w
                    losses[f'aux_query_dice_l{level_idx}'] += l_dice * w

        losses['total_loss'] = sum(losses.values())
        return losses



        return losses