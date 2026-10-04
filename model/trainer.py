import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from skimage.measure import regionprops
from omegaconf import DictConfig

from gdkvm_observability import ExperimentLogger
from gdkvm_runtime import prepare_model_runtime, resolve_runtime_policy

from utils.unified_logger import UnifiedLogger
from utils.log_integrator import Integrator
from utils.time_estimator import TimeEstimator
# from model.gdkvm01 import GDKVM
from model.gdkvm01 import GDKVM
from model.utils.parameter_groups import get_parameter_groups
from model.losses import LossComputer
from monai.metrics import (
    DiceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    ConfusionMatrixMetric,
)

log = logging.getLogger(__name__)

def compute_volume_area_length(mask_tensor: torch.Tensor) -> float:
    """
    Compute volume from mask using area-length formula.
    
    Args:
        mask_tensor: Binary mask tensor [H, W]
        
    Returns:
        Computed volume
    """
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
    
    if mask_np.sum() == 0:
        return 0.0

    props = regionprops(mask_np)
    if not props:
        return 0.0
    
    props.sort(key=lambda x: x.area, reverse=True)
    prop = props[0]

    area = prop.area
    length = prop.axis_major_length
    
    if length == 0:
        return 0.0
        
    volume = (8.0 * (area ** 2)) / (3.0 * np.pi * length)
    return volume



def _contiguous_hook(grad):
    """C1: Make gradient contiguous for AMP stability."""
    return grad.contiguous() if grad is not None else grad


class Trainer:
    """
    Training manager for video object segmentation models.
    
    Args:
        cfg: Global configuration
        stage_cfg: Stage-specific configuration (main_training/eval_stage)
        log: Unified logger instance
        run_path: Directory to save checkpoints and logs
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
    """
    def __init__(
        self,
        cfg: DictConfig,
        stage_cfg: DictConfig,
        log: UnifiedLogger,
        run_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        observer: Optional[ExperimentLogger] = None,
    ) -> None:
        self.cfg: DictConfig = cfg
        self.stage_cfg: DictConfig = stage_cfg
        self.log: UnifiedLogger = log
        self.run_path: Path = Path(run_path)
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.test_loader: DataLoader = test_loader
        self.observer: Optional[ExperimentLogger] = observer

        self.exp_id: str = cfg["exp_id"]
        self.stage: str = stage_cfg["name"]
        self.use_amp: bool = stage_cfg.amp
        self.crop_size: List[int] = stage_cfg["crop_size"]

        self.local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
        self.device: torch.device = torch.device(f"cuda:{self.local_rank}")

        self.is_distributed: bool = dist.is_available() and dist.is_initialized()
        self.rank: int = dist.get_rank() if self.is_distributed else 0
        self.world_size: int = dist.get_world_size() if self.is_distributed else 1
        self.main_process: bool = self.rank == 0
        self.runtime_policy = resolve_runtime_policy(cfg)

        # Initialize model with configuration
        model_cfg = cfg.get('model', None)
        model = GDKVM(
            model_type=model_cfg.get("model_type", "base"),
            image_encoder_type=model_cfg.get("image_encoder_type", "resnet50"),
            mask_encoder_type=model_cfg.get("mask_encoder_type", "resnet18"),
            use_channels_last=self.runtime_policy.channels_last,
            pretrained_backbones=bool(model_cfg.get("pretrained_backbones", True)),
        ).to(self.device)
        model = prepare_model_runtime(model, torch, self.runtime_policy)

        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            self.model = model


        if self.main_process:
            self.log.info(f"Runtime policy: {self.runtime_policy.as_dict()}")
            try:
                param_count = sum(p.nelement() for p in self.model.parameters()) / 1e6
                trainable_params = sum(p.nelement() for p in self.model.parameters() if p.requires_grad) / 1e6
                self.log.info(f"Model Parameters: {param_count:.2f}M (trainable: {trainable_params:.2f}M)")
            except Exception as e:
                self.log.warning(f"Model parameter count failed: {type(e).__name__}: {e}")

        self.train_integrator = Integrator(self.log, distributed=self.is_distributed)
        self._is_train = True

        parameter_groups = get_parameter_groups(
            self.model, stage_cfg, print_log=self.main_process
        )
        # C1: contiguous hook for AMP grad stability
        for group in parameter_groups:
            for param in group.get("params", []):
                if param.requires_grad:
                    param.register_hook(_contiguous_hook)
        self.optimizer = optim.AdamW(
            parameter_groups,
            lr=stage_cfg["learning_rate"],
            weight_decay=stage_cfg["weight_decay"],
            eps=1e-6 if self.use_amp else 1e-8,
            foreach=True,
        )
        self.loss_computer = LossComputer(cfg, stage_cfg)
        self.scaler = torch.amp.GradScaler(
            "cuda", init_scale=4096, enabled=self.use_amp
        )
        # B-SAFE: NaN counter for monitoring training health
        self.nan_count = 0
        self.clip_grad_norm = stage_cfg["clip_grad_norm"]

        self._init_scheduler(stage_cfg)

        self.log_text_interval = cfg.get("log_text_interval", 100)
        self.log_image_interval = cfg.get("log_image_interval", 500)
        if cfg.get("debug", False):
            self.log_text_interval = self.log_image_interval = 1

        self.log.time_estimator = TimeEstimator(
            stage_cfg.get("num_iterations", 3000), self.log_text_interval
        )

        self._init_metrics()

    def _init_scheduler(self, stage_cfg):
        if stage_cfg["lr_schedule"] == "constant":
            self.scheduler = optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0
            )
        elif stage_cfg["lr_schedule"] == "poly":
            total_num_iter = stage_cfg["num_iterations"]
            self.scheduler = optim.lr_scheduler.PolynomialLR(
                self.optimizer, total_iters=total_num_iter, power=0.9
            )
        elif stage_cfg["lr_schedule"] == "step":
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                stage_cfg["lr_schedule_steps"],
                stage_cfg["lr_schedule_gamma"],
            )
        else:
            raise NotImplementedError(
                f"Scheduler {stage_cfg['lr_schedule']} not implemented"
            )

    def _init_metrics(self):
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.iou_metric  = MeanIoU(include_background=False, reduction="mean")
        self.hd_metric   = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        self.assd_metric = SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean")

        self.conf_metric_names = [
            "precision",
            "recall",
            "accuracy",
            "specificity",
            "f1 score",
        ]
        self.conf_metric = ConfusionMatrixMetric(
            include_background=False, 
            metric_name=self.conf_metric_names,
            reduction="mean",
        )

    def _move_to_device(self, batch):
        # Normalize CPU metadata before the model/torch.compile boundary.
        # Default DataLoader collation turns per-sample Python ints into a CPU
        # tensor; leaving Tensor.item() inside GDKVM.forward creates a Dynamo
        # graph break even though this metadata is not GPU-resident.
        info = batch.get("info")
        if isinstance(info, dict) and "num_objects" in info:
            raw_num_objects = info["num_objects"]
            if isinstance(raw_num_objects, torch.Tensor):
                info["num_objects"] = [
                    int(value) for value in raw_num_objects.detach().cpu().tolist()
                ]
            elif isinstance(raw_num_objects, (list, tuple)):
                info["num_objects"] = [
                    int(value.item()) if isinstance(value, torch.Tensor) else int(value)
                    for value in raw_num_objects
                ]
            else:
                info["num_objects"] = [int(raw_num_objects)]

        # B-SAFE: only image features get channels_last memory format
        # Labels (cls_gt, masks, etc.) are not used in conv ops, no need
        CHANNELS_LAST_KEYS = {"rgb", "pixfeat"}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                continue
            if self.runtime_policy.channels_last and key in CHANNELS_LAST_KEYS and value.ndim == 4:
                batch[key] = value.to(
                    self.device,
                    memory_format=torch.channels_last,
                    non_blocking=True,
                )
            elif self.runtime_policy.channels_last and key in CHANNELS_LAST_KEYS and value.ndim == 5:
                batch[key] = value.to(
                    self.device,
                    memory_format=torch.channels_last_3d,
                    non_blocking=True,
                )
            else:
                batch[key] = value.to(self.device, non_blocking=True)
        return batch

    def _ensure_finite_outputs(self, outputs):
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and (
                    value.is_floating_point() or value.is_complex()
                ):
                    # Avoid a Python-side bool check on torch.isfinite(...).all().
                    # On accelerator tensors that check synchronizes the device
                    # once per output. nan_to_num is an identity on finite values
                    # and preserves the existing non-finite replacement contract.
                    outputs[key] = torch.nan_to_num(
                        value, nan=0.0, posinf=1e4, neginf=-1e4
                    )
        return outputs

    def train(self):
        self._is_train = True
        self.model.train()
        return self

    def val(self):
        self._is_train = False
        self.model.eval()
        return self

    def do_pass(self, data: Dict[str, Any], it: int = 0) -> torch.Tensor:
        """
        Perform a single training/evaluation pass.
        
        Args:
            data: Batch data dictionary containing 'rgb', 'cls_gt', etc.
            it: Current iteration number
            
        Returns:
            Loss tensor
        """
        torch.set_grad_enabled(self._is_train)
        self._move_to_device(data)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            out = self.model(data)
            out = self._ensure_finite_outputs(out)

            num_objects = out.get("num_objects", [1] * data["rgb"].shape[0])
            data.update(out)

            T = data["rgb"].shape[1]
            supervised_frame_indices = (0, T - 1)
            all_logits_keys = [f"logits_{ti}" for ti in supervised_frame_indices]
            if not all(k in data for k in all_logits_keys):
                raise KeyError(
                    f"Missing logits keys. Expected {all_logits_keys}, found {list(data.keys())}"
                )

            new_pred = torch.stack([data[k] for k in all_logits_keys], dim=1)
            new_gt = torch.stack(
                (data["cls_gt"][:, 0], data["cls_gt"][:, -1]),
                dim=1,
            )
            data.update({"logits": new_pred, "masks": new_gt})

            losses = self.loss_computer.compute(data, num_objects)
            loss = losses["total_loss"]

        if not torch.isfinite(loss):
            self.nan_count += 1
            if self.main_process and self.nan_count % 100 == 0:
                self.log.warning(
                    f"[Trainer] NaN count = {self.nan_count} at iter {it}"
                )
            return torch.tensor(0.0, device=self.device)

        self.optimizer.zero_grad(set_to_none=True)

        self.scaler.scale(loss).backward()

        if self.clip_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        if it % self.log_text_interval == 0:
            loss_val = loss.detach().item()
            lr_val = self.scheduler.get_last_lr()[0]
            self.log.log_scalar("loss", loss_val, it, to_wandb=False)
            if it != 0:
                self.log.log_scalar("lr", lr_val, it, to_wandb=False)

            if self.main_process and self.observer is not None:
                components = {}
                for key, value in losses.items():
                    if key == "total_loss":
                        continue
                    if isinstance(value, torch.Tensor):
                        components[key] = value.detach().item()
                    elif isinstance(value, (int, float)):
                        components[key] = value
                self.observer.log_train_step(
                    step=it,
                    total_loss=loss_val,
                    lr=lr_val,
                    loss_components=components,
                    nonfinite_count=self.nan_count,
                )

        return loss.detach()

    def evaluate(
        self, 
        val_loader: DataLoader, 
        epoch: int, 
        run_path: str, 
        it: int, 
        local_rank: Optional[int] = None, 
        world_size: Optional[int] = None
    ) -> Dict[str, float]:
        """Run evaluation on validation set."""
        return self._run_evaluation(val_loader, "val", epoch, run_path, it)

    def test(
        self, 
        test_loader: DataLoader, 
        epoch: int, 
        run_path: str, 
        it: int, 
        local_rank: Optional[int] = None, 
        world_size: Optional[int] = None
    ) -> Dict[str, float]:
        """Run evaluation on test set."""
        return self._run_evaluation(test_loader, "test", epoch, run_path, it)

    def _reset_metrics(self):
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.hd_metric.reset()
        self.assd_metric.reset()
        self.conf_metric.reset()

    def _run_evaluation(
        self, 
        data_loader: DataLoader, 
        mode: str, 
        epoch: int, 
        run_path: str, 
        it: int
    ) -> Dict[str, float]:
        if self.is_distributed:
            dist.barrier()

        if self.main_process:
            self.log.info(
                f"[{mode.capitalize()}] Iter {it} Epoch {epoch}: Start Evaluation..."
            )

        prev_mode = self.model.training
        self.model.eval()
        self._reset_metrics()

        val_ef_preds = []
        val_ef_gts = []

        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                self._move_to_device(batch_data)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = self.model(batch_data)
                    out = self._ensure_finite_outputs(out)

                T = batch_data["rgb"].shape[1]
                mask_keys = [f"masks_{0}", f"masks_{T - 1}"]

                if not all(k in out for k in mask_keys):
                    continue

                masks_first = out[mask_keys[0]]
                masks_last = out[mask_keys[1]]

                if masks_first.shape[1] > 1:
                    masks_first = masks_first[:, 1:2, ...]
                    masks_last = masks_last[:, 1:2, ...]

                gt = batch_data["cls_gt"]
                if gt.dim() == 5:
                    gt = gt.squeeze(2)

                gt_first = gt[:, 0, ...].unsqueeze(1)
                gt_last = gt[:, -1, ...].unsqueeze(1)

                preds_concat = torch.cat([masks_first, masks_last], dim=0)
                gts_concat = torch.cat([gt_first, gt_last], dim=0)

                preds_bin = (preds_concat > 0.5).float()
                gts_bin = (gts_concat > 0.5).float()

                self.dice_metric(y_pred=preds_bin, y=gts_bin)
                self.iou_metric(y_pred=preds_bin, y=gts_bin)
                self.hd_metric(y_pred=preds_bin, y=gts_bin)
                self.assd_metric(y_pred=preds_bin, y=gts_bin)
                self.conf_metric(y_pred=preds_bin, y=gts_bin)

                B = masks_first.shape[0]
                
                current_ef_preds = []
                current_ef_gts = []
                
                for b in range(B):
                    pred_ed_mask = preds_bin[b, 0]
                    pred_es_mask = preds_bin[b + B, 0]
                    
                    gt_ed_mask = gts_bin[b, 0]
                    gt_es_mask = gts_bin[b + B, 0]
                    
                    vol_ed_pred = compute_volume_area_length(pred_ed_mask)
                    vol_es_pred = compute_volume_area_length(pred_es_mask)
                    
                    vol_ed_gt = compute_volume_area_length(gt_ed_mask)
                    vol_es_gt = compute_volume_area_length(gt_es_mask)
                    
                    ef_pred_val = (vol_ed_pred - vol_es_pred) / (vol_ed_pred + 1e-6)
                    ef_gt_val = (vol_ed_gt - vol_es_gt) / (vol_ed_gt + 1e-6)
                    
                    current_ef_preds.append(ef_pred_val)
                    current_ef_gts.append(ef_gt_val)

                batch_ef_pred_tensor = torch.tensor(current_ef_preds, dtype=torch.float32, device=self.device)
                batch_ef_gt_tensor = torch.tensor(current_ef_gts, dtype=torch.float32, device=self.device)

                val_ef_preds.append(batch_ef_pred_tensor)
                val_ef_gts.append(batch_ef_gt_tensor)

                vis_limit = self.stage_cfg.get("num_vis", 0)

                if self.main_process and batch_idx < vis_limit:
                    self._visualize_batch(batch_data, out, batch_idx, it, epoch, mode)

        local_counts = len(self.dice_metric.get_buffer())

        metrics_map = {
            "dice": self.dice_metric,
            "iou": self.iou_metric,
            "hd95": self.hd_metric,
            "assd": self.assd_metric,
        }

        local_results = {}
        for name, metric in metrics_map.items():
            try:
                res = metric.aggregate()
                if isinstance(res, torch.Tensor):
                    res = res.item()
                local_results[name] = res if np.isfinite(res) else 0.0
            except (RuntimeError, ValueError, AttributeError) as e:
                # Metric computation errors (empty buffer, wrong shape, etc.)
                self.log.debug(f"Metric '{name}' aggregation failed: {e}")
                local_results[name] = 0.0
            except Exception as e:
                # Unexpected errors
                self.log.warning(f"Unexpected error in metric '{name}': {type(e).__name__}: {e}")
                local_results[name] = 0.0

        try:
            conf_res = self.conf_metric.aggregate()
            conf_names = ["precision", "recall", "acc", "sp", "F1"]
            # C4: Look up by name (not position) for MONAI version compat
            if isinstance(conf_res, tuple) and len(conf_res) == 2:
                conf_values, conf_idx = conf_res
                for name in conf_names:
                    if name in conf_idx:
                        val = conf_values[conf_idx.index(name)].item()
                    else:
                        val = 0.0
                    local_results[name] = val if np.isfinite(val) else 0.0
            else:
                for i, name in enumerate(conf_names):
                    val = conf_res[i].item()
                    local_results[name] = val if np.isfinite(val) else 0.0
        except (RuntimeError, ValueError, IndexError) as e:
            # Confusion matrix metric errors
            self.log.debug(f"Confusion matrix metric aggregation failed: {e}")
        except Exception as e:
            self.log.warning(f"Unexpected error in confusion metrics: {type(e).__name__}: {e}")

        global_metrics = self._reduce_metrics_weighted(local_results, local_counts)

        if len(val_ef_preds) > 0:
            local_ef_pred = torch.cat(val_ef_preds, dim=0)
            local_ef_gt = torch.cat(val_ef_gts, dim=0)
        else:
            local_ef_pred = torch.tensor([], device=self.device)
            local_ef_gt = torch.tensor([], device=self.device)

        global_ef_pred = self._gather_all_tensors(local_ef_pred)
        global_ef_gt = self._gather_all_tensors(local_ef_gt)

        if self.main_process:
            if global_ef_pred.numel() > 1:
                ef_p = global_ef_pred.cpu().numpy()
                ef_g = global_ef_gt.cpu().numpy()

                valid_mask = np.isfinite(ef_p) & np.isfinite(ef_g)
                if valid_mask.sum() > 1:
                    corr = np.corrcoef(ef_p[valid_mask], ef_g[valid_mask])[0, 1]
                    diff = ef_p[valid_mask] - ef_g[valid_mask]
                    bias, std = diff.mean(), diff.std()
                    mae = np.mean(np.abs(diff))
                else:
                    corr, bias, std, mae = 0.0, 0.0, 0.0, 0.0

                global_metrics["lvef_corr"] = corr
                global_metrics["lvef_bias"] = bias
                global_metrics["lvef_std"] = std
                global_metrics["lvef_mae"] = mae
            else:
                global_metrics["lvef_corr"] = 0.0
                global_metrics["lvef_bias"] = 0.0
                global_metrics["lvef_std"] = 0.0

            self._log_final_metrics(global_metrics, mode, it, epoch)

        if self.is_distributed:
            dist.barrier()

        self.model.train(prev_mode)
        return global_metrics

    def _visualize_batch(self, batch_data, out, batch_idx, it, epoch, mode):
        try:
            # Visualization is an optional diagnostic dependency. Import lazily
            # so core Trainer/model usage does not require matplotlib.
            from vis.vis_0730 import visualize_sequence

            rgb_seq = batch_data["rgb"][0].cpu().numpy()
            cls_gt_seq = batch_data["cls_gt"][0].cpu().numpy()

            patient_name = f"b{batch_idx}"
            if "info" in batch_data and "name" in batch_data["info"]:
                patient_name = str(batch_data["info"]["name"][0])

            visualize_sequence(
                rgb_seq,
                cls_gt_seq,
                out,
                str(self.run_path),
                f"vis_idx_{batch_idx}",
                iteration=it,
                epoch=epoch,
                patient_id=patient_name,
                mode=mode,
            )
        except Exception as e:
            self.log.warning(f"Vis failed: {e}")

    def _reduce_metrics_weighted(self, local_metrics: dict, local_count: int):
        if not self.is_distributed:
            return local_metrics

        keys = list(local_metrics.keys())
        local_vec = [float(local_count)]
        for k in keys:
            local_vec.append(local_metrics[k] * local_count)

        tensor_vec = torch.tensor(local_vec, device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor_vec, op=dist.ReduceOp.SUM)

        total_count = tensor_vec[0].item()
        global_metrics = {}

        if total_count > 0:
            for i, k in enumerate(keys):
                global_metrics[k] = tensor_vec[i + 1].item() / total_count
        else:
            for k in keys:
                global_metrics[k] = 0.0

        return global_metrics

    def _gather_all_tensors(self, local_tensor):
        if not self.is_distributed:
            return local_tensor

        local_size = torch.tensor(
            [local_tensor.numel()], dtype=torch.long, device=self.device
        )
        size_list = [
            torch.tensor([0], dtype=torch.long, device=self.device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(size_list, local_size)

        max_size = max([s.item() for s in size_list])
        if max_size == 0:
            return torch.tensor([], device=self.device)

        padded = torch.zeros(max_size, dtype=local_tensor.dtype, device=self.device)
        if local_tensor.numel() > 0:
            padded[: local_tensor.numel()] = local_tensor

        gathered_list = [torch.zeros_like(padded) for _ in range(self.world_size)]
        dist.all_gather(gathered_list, padded)

        final_data = []
        for i, size_tensor in enumerate(size_list):
            actual_size = size_tensor.item()
            if actual_size > 0:
                final_data.append(gathered_list[i][:actual_size])

        return (
            torch.cat(final_data)
            if final_data
            else torch.tensor([], device=self.device)
        )

    def _log_final_metrics(self, metrics, mode, it, epoch):
        log_items = []
        for k, v in metrics.items():
            if "lvef" in k:
                log_items.append(f"{k.upper()}={v:.4f}")
            else:
                log_items.append(f"{k.upper()}={v:.4f}")
        
        log_str = f"[{mode.capitalize()}] Iter={it} | " + " | ".join(log_items)
        self.log.info(log_str)

        # 通过 UnifiedLogger 记录到 WandB

        # Debug: print dice to stdout for real-time monitoring
        dice_val = metrics.get("dice", 0.0)
        print(f"[DEBUG] Iter={it} dice={dice_val:.4f}", flush=True)
        # The imported evaluator predates gdkvm-rerelease-v1. Keep its
        # projections explicitly namespaced as legacy until the protocol-owned
        # evaluator replaces these calculations.
        legacy_metrics = {f"legacy/{k.lower()}": v for k, v in metrics.items()}
        self.log.log_scalars(
            {f"{mode}/{key}": value for key, value in legacy_metrics.items()},
            it,
            to_tb=False,
            to_wandb=False,
        )
        if self.main_process and self.observer is not None:
            self.observer.log_eval_summary(
                mode, step=it, epoch=epoch, metrics=legacy_metrics
            )

    def _model_for_state(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def save_checkpoint(self, it: int, keep_last_n: int = 3):
        """
        Save checkpoint and optionally cleanup old ones.
        
        Args:
            it: Current iteration number
            keep_last_n: Number of recent checkpoints to keep. 
                        Set to 0 to keep all checkpoints.
        """
        if not self.main_process:
            return
        self.run_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.run_path / f"{self.exp_id}_{self.stage}_ckpt_{it}.pth"

        torch.save(
            {
                "it": it,
                "model": self._model_for_state().state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            ckpt_path,
        )
        self.log.info(f"Saved checkpoint: {ckpt_path}")
        
        # Cleanup old checkpoints
        if keep_last_n > 0:
            self._cleanup_old_checkpoints(keep_last_n)
        return ckpt_path

    def _cleanup_old_checkpoints(self, keep_last_n: int) -> None:
        """
        Remove old checkpoint files, keeping only the most recent N.
        
        Args:
            keep_last_n: Number of checkpoints to preserve
        """
        pattern = f"{self.exp_id}_{self.stage}_ckpt_*.pth"
        ckpt_files = sorted(
            self.run_path.glob(pattern), 
            key=lambda x: x.stat().st_mtime
        )
        
        if len(ckpt_files) <= keep_last_n:
            return
            
        for old_ckpt in ckpt_files[:-keep_last_n]:
            try:
                old_ckpt.unlink()
                self.log.info(f"Removed old checkpoint: {old_ckpt.name}")
            except OSError as e:
                self.log.warning(f"Failed to remove old checkpoint {old_ckpt}: {e}")

    def load_checkpoint(self, path: str) -> int:
        """
        Load checkpoint from file.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Iteration number of the checkpoint
        """
        self.log.info(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self._model_for_state().load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        return ckpt["it"]