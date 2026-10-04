"""Compatibility facade for Python logging and TensorBoard.

Remote experiment projection is owned by gdkvm_observability.ExperimentLogger.
The legacy ``enabled_wandb``/``to_wandb`` arguments remain accepted so imported
call sites fail softly while they are migrated, but this facade never imports
or calls W&B.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Union

import numpy as np
import torch

from utils.logger import TensorboardLogger


class UnifiedLogger:
    def __init__(
        self,
        run_dir: str,
        py_logger: Optional[logging.Logger] = None,
        *,
        enabled_tb: bool = True,
        enabled_wandb: bool = False,
        git_info_enabled: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self._py_log = py_logger or logging.getLogger(__name__)
        if enabled_wandb:
            self._py_log.warning(
                "enabled_wandb is deprecated here; use ExperimentLogger/WandbSink"
            )
        self._tb_logger = TensorboardLogger(
            run_dir,
            self._py_log,
            enabled_tb=enabled_tb,
            git_info_enabled=git_info_enabled,
        )

    def log_scalar(
        self,
        tag: str,
        value: Union[float, int, torch.Tensor],
        step: int,
        *,
        to_tb: bool = True,
        to_wandb: bool = False,
    ) -> None:
        del to_wandb
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        if to_tb:
            self._tb_logger.log_scalar(tag, float(value), step)

    def log_scalars(
        self,
        scalars: Dict[str, Union[float, int, torch.Tensor]],
        step: int,
        *,
        prefix: str = "",
        to_tb: bool = True,
        to_wandb: bool = False,
    ) -> None:
        for tag, value in scalars.items():
            full_tag = f"{prefix}/{tag}" if prefix else tag
            self.log_scalar(full_tag, value, step, to_tb=to_tb, to_wandb=to_wandb)

    def log_metrics(
        self,
        exp_id: str,
        prefix: str,
        metrics: Dict,
        step: int,
        *,
        to_tb: bool = True,
        to_wandb: bool = False,
        to_console: bool = True,
    ) -> None:
        del to_wandb
        if to_console:
            self._tb_logger.log_metrics(exp_id, prefix, metrics, step)
        elif to_tb:
            for key, value in metrics.items():
                self.log_scalar(f"{prefix}/{key}", value, step, to_tb=True)

    def log_image(
        self,
        stage_name: str,
        tag: str,
        image: Union[np.ndarray, torch.Tensor],
        step: int,
        *,
        to_tb: bool = True,
        to_wandb: bool = False,
        wandb_caption: Optional[str] = None,
    ) -> None:
        del to_wandb, wandb_caption
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(2)
        if to_tb:
            self._tb_logger.log_image(stage_name, tag, image, step)

    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None,
        *,
        to_tb: bool = True,
        to_wandb: bool = False,
    ) -> None:
        del step, to_wandb
        if to_tb:
            self._tb_logger.log_string(tag, text)
        else:
            self._py_log.info("%s - %s", tag, text)

    def debug(self, msg: str) -> None:
        self._py_log.debug(msg)

    def info(self, msg: str) -> None:
        self._py_log.info(msg)

    def warning(self, msg: str) -> None:
        self._py_log.warning(msg)

    def error(self, msg: str) -> None:
        self._py_log.error(msg)

    @property
    def time_estimator(self):
        return self._tb_logger.time_estimator

    @time_estimator.setter
    def time_estimator(self, value) -> None:
        self._tb_logger.time_estimator = value

    def close(self) -> None:
        if self._tb_logger.tb_log is not None:
            self._tb_logger.tb_log.close()
