#!/usr/bin/env python3
"""
DiffCorn MLP 模型 - 支持相机条件化的双模态输入

版本:
  - 原版: 1024 维 DINOv3 特征 -> 64 维 ADEL 物理参数
  - 相机条件化: (1024 维 DINOv3 + 1 维相机高度) -> 64 维 ADEL 物理参数

架构升级:
  Input(1025) -> Linear(1025,512) -> BN -> ReLU
           -> Linear(512,256) -> BN -> ReLU
           -> Linear(256,64) -> Sigmoid -> Scale
"""

import torch
import torch.nn as nn


class DiffCornMLP(nn.Module):
    """
    多层感知机：将 DINOv3 特征（可选 + 相机高度）映射到 64 维 ADEL 物理参数

    相机条件化版本:
        - 视觉输入: 1024 维 DINOv3 特征
        - 物理输入: 1 维归一化相机高度 (cam_h / 200.0)
        - 融合策略: 拼接后得到 1025 维输入

    架构:
        Input(1024 or 1025) -> Linear(->512) -> BN -> ReLU
                         -> Linear(512,256) -> BN -> ReLU
                         -> Linear(256,64) -> Sigmoid -> Scale

    输出 64 维参数对应 16 层 Phytomer，每层 4 个参数:
        - 节间长度 (Internode length): 0-40 cm
        - 叶片角度 (Leaf angle): 0-360 度
        - 叶片长度 (Leaf length): 0-100 cm
        - 叶片倾角 (Leaf inclination): 0-90 度
    """

    def __init__(self, input_dim: int = 1024, hidden_dims: list = [512, 256, 64],
                 use_camera_condition: bool = False, cam_h_scale: float = 200.0):
        """
        Args:
            input_dim: DINOv3 特征维度 (默认 1024)
            hidden_dims: 隐藏层维度列表
            use_camera_condition: 是否使用相机条件化（添加 cam_h 作为输入）
            cam_h_scale: 相机高度归一化系数（默认 200.0）
        """
        super(DiffCornMLP, self).__init__()

        self.use_camera_condition = use_camera_condition
        self.cam_h_scale = cam_h_scale

        # 如果使用相机条件化，输入维度增加 1
        if use_camera_condition:
            input_dim += 1  # 1024 -> 1025

        self.input_dim = input_dim

        # 构建网络层
        layers = []
        in_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim

        # 输出层（没有激活，后续会用自定义的 Sigmoid + Scale）
        layers.append(nn.Linear(in_dim, hidden_dims[-1]))

        self.network = nn.Sequential(*layers)

        # 定义输出范围（ADEL 物理参数的合理范围）
        # 64 维 = 16 层 × 4 参数/层
        # 每层的 4 个参数: [internode, angle, leaf_len, leaf_inclination]
        self.register_buffer('output_min', torch.zeros(64))
        self.register_buffer('output_max', torch.ones(64) * 40.0)  # 默认最大 40cm

        # 更精确地设置每层参数的范围
        for i in range(16):
            # 节间长度: 0-40 cm
            self.output_min[i * 4 + 0] = 0.0
            self.output_max[i * 4 + 0] = 40.0

            # 叶片角度: 0-360 度
            self.output_min[i * 4 + 1] = 0.0
            self.output_max[i * 4 + 1] = 360.0

            # 叶片长度: 0-100 cm
            self.output_min[i * 4 + 2] = 0.0
            self.output_max[i * 4 + 2] = 100.0

            # 叶片倾角: 0-90 度
            self.output_min[i * 4 + 3] = 0.0
            self.output_max[i * 4 + 3] = 90.0

    def forward(self, x, cam_h=None):
        """
        前向传播

        Args:
            x: (batch_size, 1024) DINOv3 特征
            cam_h: (batch_size, 1) 或 (batch_size,) 相机高度 [cm]，仅当 use_camera_condition=True 时需要

        Returns:
            (batch_size, 64) ADEL 物理参数，范围钳制在物理合理区间
        """
        # 相机条件化：拼接归一化的相机高度
        if self.use_camera_condition:
            if cam_h is None:
                raise ValueError("cam_h must be provided when use_camera_condition=True")

            # 归一化相机高度
            if cam_h.dim() == 1:
                cam_h = cam_h.unsqueeze(1)
            cam_h_normalized = cam_h / self.cam_h_scale  # 归一化到 [0, 1] 范围

            # 拼接 DINOv3 特征和相机高度
            x = torch.cat([x, cam_h_normalized], dim=1)  # (batch_size, 1025)

        # 通过 MLP
        x = self.network(x)  # (batch_size, 64)

        # Sigmoid 钳制到 [0, 1]
        x = torch.sigmoid(x)

        # 缩放到物理范围
        output = x * (self.output_max - self.output_min) + self.output_min

        return output

    def get_phytomer_params(self, output, layer_idx: int):
        """
        从输出中提取指定 Phytomer 层的参数

        Args:
            output: (batch_size, 64) 模型输出
            layer_idx: 层索引 (0-15)

        Returns:
            dict: {internode, angle, leaf_length, leaf_inclination}
        """
        assert 0 <= layer_idx < 16, "layer_idx must be in [0, 15]"

        params = {
            'internode': output[:, layer_idx * 4 + 0],      # 节间长度 (cm)
            'angle': output[:, layer_idx * 4 + 1],          # 叶片角度 (度)
            'leaf_length': output[:, layer_idx * 4 + 2],    # 叶片长度 (cm)
            'leaf_inclination': output[:, layer_idx * 4 + 3],     # 叶片倾角 (度)
        }
        return params

    def print_architecture(self):
        """打印网络架构信息"""
        print("=" * 60)
        if self.use_camera_condition:
            print("DiffCornMLP Architecture (Camera-Conditioned)")
        else:
            print("DiffCornMLP Architecture (Original)")
        print("=" * 60)

        if self.use_camera_condition:
            print(f"Input:  1024 dim (DINOv3 ViT-L features) + 1 dim (camera height)")
            print(f"        -> 1025 dim (concatenated)")
            print(f"Camera height normalization: / {self.cam_h_scale}")
        else:
            print(f"Input:  1024 dim (DINOv3 ViT-L features)")

        print(f"Hidden: 512 -> 256 (with BatchNorm + ReLU)")
        print(f"Output: 64 dim (16 Phytomers × 4 params)")
        print(f"\nOutput ranges:")
        print(f"  - Internode length: 0-40 cm")
        print(f"  - Leaf angle:       0-360 deg")
        print(f"  - Leaf length:      0-100 cm")
        print(f"  - Leaf inclination:       0-90 deg")
        print("=" * 60)


if __name__ == "__main__":
    # 测试模型
    model = DiffCornMLP()
    model.print_architecture()

    # 随机输入
    batch_size = 4
    x = torch.randn(batch_size, 1024)

    # 前向传播
    output = model(x)

    print(f"\nTest input shape:  {x.shape}")
    print(f"Test output shape: {output.shape}")

    # 检查输出范围
    print(f"\nOutput range check:")
    print(f"  Min: {output.min().item():.2f}")
    print(f"  Max: {output.max().item():.2f}")
    print(f"  Mean: {output.mean().item():.2f}")

    # 提取第一层参数
    params = model.get_phytomer_params(output, layer_idx=0)
    print(f"\nFirst phytomer layer params (sample 0):")
    for k, v in params.items():
        print(f"  {k}: {v[0].item():.2f}")
