#!/usr/bin/env python3
"""
Domain-adversarial wrapper for DiffCornMLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function

from model import DiffCornMLP


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor, lambda_: float | None = None) -> torch.Tensor:
        coeff = self.lambda_ if lambda_ is None else float(lambda_)
        return GradientReversalFunction.apply(x, coeff)


class FeatureAdapter(nn.Module):
    """
    Minimal trainable feature adapter so adversarial gradients have a learnable path.
    """

    def __init__(self, dim: int = 1024) -> None:
        super().__init__()
        self.delta = nn.Linear(dim, dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.delta(x)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffCornDANN(nn.Module):
    def __init__(self, use_camera_condition: bool = True, cam_h_scale: float = 200.0) -> None:
        super().__init__()
        self.adapter = FeatureAdapter(dim=1024)
        self.regressor = DiffCornMLP(input_dim=1024, use_camera_condition=use_camera_condition, cam_h_scale=cam_h_scale)
        self.domain_classifier = DomainClassifier(input_dim=1024)
        self.grl = GradientReversal()

    @property
    def use_camera_condition(self) -> bool:
        return self.regressor.use_camera_condition

    def adapted_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.adapter(features)

    def regress(self, adapted_features: torch.Tensor, cam_h: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_camera_condition:
            return self.regressor(adapted_features, cam_h)
        return self.regressor(adapted_features)

    def domain_predict(self, adapted_features: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        rev = self.grl(adapted_features, grl_lambda)
        return self.domain_classifier(rev)

    def forward(
        self,
        features: torch.Tensor,
        cam_h: torch.Tensor | None = None,
        grl_lambda: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        adapted = self.adapted_features(features)
        outputs = self.regress(adapted, cam_h)
        domain_probs = self.domain_predict(adapted, grl_lambda)
        return outputs, domain_probs, adapted
