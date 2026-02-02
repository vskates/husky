from __future__ import annotations
from dataclasses import dataclass
import torch

from .trainer import Trainer


@dataclass
class GraspConfig:
    num_rotations: int = 16
    force_cpu: bool = False
    weights_path: str = ""


class GraspModel:
    def __init__(self, cfg: GraspConfig):
        self.cfg = cfg
        self.trainer = Trainer(force_cpu=cfg.force_cpu, num_rotations=cfg.num_rotations)
        if cfg.weights_path:
            self.trainer.load(cfg.weights_path)
        self.trainer.model.eval()

    @torch.no_grad()
    def infer_q(self, color_hm_u8: torch.Tensor, height_hm_f32: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
          color_hm_u8: (H,W,3) uint8 tensor
          height_hm_f32: (H,W) float32 tensor
        Returns:
          Q: (K,H,W) float tensor
        """
        # важно: чтобы выбиралась ориентация, надо specific_rotation=-1 (все углы)
        q, *_ = self.trainer.forward(color_hm_u8, height_hm_f32, specific_rotation=-1)
        return q
