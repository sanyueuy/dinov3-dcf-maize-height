#!/usr/bin/env python3
"""
DiffCorn-Fusion 训练脚本 - 手绘 BBOX 特征 + 相机条件化

配置:
- 数据：training_bundle_hand.pt (156 样本，手绘标注)
- 模型：相机条件化 MLP (1025D 输入 = 1024 特征 + 1 相机高度)
- 策略：混合尺度训练 (混合 106cm 和 140cm 数据)
- 输出：64D Phytomer 参数 (16 节 × 4 参数)
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import DiffCornMLP

# ==================== 配置 ====================
CONFIG = {
    # 数据路径
    'bundle_path': 'training_bundle_hand.pt',  # 手绘 BBOX 特征包

    # ADEL 先验配置 (Synthetic-to-Real Prior Constraint)
    'use_adel_prior': True,           # 是否使用 ADEL 先验约束
    'adel_prior_path': r'D:\cornTrain\DINOV3_retrain_clean\data\adel\adel_tall_maize_prior_stats.npz',  # ADEL 统计文件
    'prior_loss_weight': 0.01,        # 先验损失权重 λ
    'prior_loss_type': 'l2',          # 'l2' 或 'mahalanobis'

    # 模型配置
    'use_camera_condition': True,  # 相机条件化
    'cam_h_scale': 200.0,  # 相机高度归一化系数

    # 训练参数
    'batch_size': 16,
    'num_epochs': 300,
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'delta': 1.0,  # HuberLoss delta

    # 输出路径
    'output_dir': './checkpoints',
    'best_model_name': None,
    'final_model_name': None,
    'report_name': 'diffcorn_fusion_hand_report.txt',
    'curve_name': 'diffcorn_fusion_hand_training_curves.png',
    'print_freq': 20,

    # 随机种子
    'seed': 42,
}


def load_adel_prior(prior_path: str, device: torch.device):
    path = Path(prior_path)
    if not path.exists():
        raise FileNotFoundError(f"ADEL prior not found: {prior_path}")

    if path.suffix.lower() == '.npz':
        data = np.load(path, allow_pickle=True)
        return {
            'mean': torch.tensor(data['mean'], dtype=torch.float32, device=device),
            'std': torch.tensor(data['std'], dtype=torch.float32, device=device),
            'inv_cov': torch.tensor(data['inv_cov'], dtype=torch.float32, device=device),
            'param_cols': [str(x) for x in data['param_cols'].tolist()],
            'source_path': str(path),
            'source_format': 'npz',
        }

    prior = torch.load(path, map_location=device)
    prior['mean'] = prior['mean'].to(device)
    prior['std'] = prior['std'].to(device)
    prior['inv_cov'] = prior['inv_cov'].to(device)
    prior['source_path'] = str(path)
    prior['source_format'] = 'pt'
    return prior


# ==================== 数据集类 ====================
class HandBboxFusionDataset(Dataset):
    """手绘 BBOX 特征数据集 - 支持相机条件化"""

    def __init__(self, bundle_path: str):
        # 加载 bundle
        bundle = torch.load(bundle_path)
        self.features = bundle['features']           # (N, 1024)
        self.heights = bundle['heights']             # (N,) 真实高度
        self.pixel_heights = bundle['pixel_heights'] # (N,) 像素高度
        self.cam_heights = bundle['cam_heights']     # (N,) 相机高度
        self.filenames = bundle['filenames']

        print(f"Loaded {len(self.features)} samples")

        # 统计相机高度分布
        cam_106 = sum(1 for h in self.cam_heights if abs(h - 106) < 1)
        cam_140 = sum(1 for h in self.cam_heights if abs(h - 140) < 1)
        print(f"  Camera 106cm: {cam_106} samples")
        print(f"  Camera 140cm: {cam_140} samples")

        # 统计真实高度分布
        print(f"  Height range: {self.heights.min():.1f} - {self.heights.max():.1f} cm")
        print(f"  Height mean:  {self.heights.mean():.1f} cm")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],           # (1024,)
            'cam_height': torch.tensor([self.cam_heights[idx]], dtype=torch.float32),  # (1,)
            'target': torch.tensor([self.heights[idx]], dtype=torch.float32),  # (1,) 真实高度
            'filename': self.filenames[idx],
        }


# ==================== 评估函数 ====================
def evaluate(model, dataloader, device, use_camera_condition=True):
    """计算 MAE"""
    model.eval()
    total_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            cam_heights = batch['cam_height'].to(device)

            # 前向传播
            if use_camera_condition:
                outputs = model(features, cam_heights.squeeze(-1))
            else:
                outputs = model(features)

            # 从 64D phytomer 参数计算预测高度
            # 方法：取 internode_length 之和
            # 64D = 16 phytomers × 4 params
            # params: [internode, azimuth, leaf_length, inclination]
            internode_lengths = outputs[:, 0::4]  # 每 4 个取第 1 个 (16 个节间长度)
            predicted_height = internode_lengths.sum(dim=1, keepdim=True)

            error = torch.abs(predicted_height - targets).sum().item()
            total_error += error
            total_samples += targets.size(0)

    mae = total_error / total_samples
    return mae


def evaluate_detailed(model, dataloader, device, use_camera_condition=True):
    """详细评估 - 返回预测值和真实值"""
    model.eval()
    all_preds = []
    all_targets = []
    all_filenames = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            cam_heights = batch['cam_height'].to(device)
            filenames = batch['filename']

            # 前向传播
            if use_camera_condition:
                outputs = model(features, cam_heights.squeeze(-1))
            else:
                outputs = model(features)

            # 从 64D phytomer 参数计算预测高度
            internode_lengths = outputs[:, 0::4]
            predicted_height = internode_lengths.sum(dim=1, keepdim=True)

            all_preds.extend(predicted_height.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            all_filenames.extend(filenames)

    return np.array(all_preds), np.array(all_targets), all_filenames


# ==================== 训练函数 ====================
def train_diffcorn_fusion():
    """训练 DiffCorn-Fusion 模型"""

    print("=" * 70)
    print("DiffCorn-Fusion Training - Hand BBOX + Camera Condition")
    print("=" * 70)

    # 设置随机种子
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # 1. 加载数据
    print("\n[1/6] Loading hand-annotated BBOX data...")
    dataset = HandBboxFusionDataset(CONFIG['bundle_path'])

    # 混合尺度训练：不进行划分，使用全部数据
    # 随机打乱
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(0.8 * len(dataset))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    # 保存测试集索引用于后续评估
    test_filenames = [dataset.filenames[i] for i in test_indices]
    test_indices_data = {
        'test_indices': test_indices,
        'test_filenames': test_filenames,
        'train_size': train_size,
        'test_size': len(test_indices),
        'seed': CONFIG['seed'],
    }
    indices_path = os.path.join(CONFIG['output_dir'], 'hand_bbox_test_indices.json')
    with open(indices_path, 'w') as f:
        json.dump(test_indices_data, f, indent=2)
    print(f"Saved test indices: {indices_path}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
    )

    # 2. 创建模型
    print("\n[2/6] Creating DiffCorn-Fusion model...")
    model = DiffCornMLP(
        input_dim=1024,
        use_camera_condition=CONFIG['use_camera_condition'],
        cam_h_scale=CONFIG['cam_h_scale']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.print_architecture()
    print(f"Device: {device}")

    # 3. 损失函数和优化器
    print("\n[3/6] Setting up loss and optimizer...")
    criterion = nn.HuberLoss(delta=CONFIG['delta'])

    # 加载 ADEL 先验统计
    adel_prior = None
    if CONFIG['use_adel_prior']:
        prior_path = CONFIG['adel_prior_path']
        if os.path.exists(prior_path):
            adel_prior = load_adel_prior(prior_path, device)
            print(f"  ADEL prior loaded: {prior_path}")
            print(f"  Prior format: {adel_prior.get('source_format', 'unknown')}")
            print(f"  Prior loss type: {CONFIG['prior_loss_type']}")
            print(f"  Prior loss weight (lambda): {CONFIG['prior_loss_weight']}")
        else:
            print(f"  Warning: ADEL prior not found at {prior_path}, skipping prior constraint")
            CONFIG['use_adel_prior'] = False

    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['num_epochs'],
        eta_min=1e-6
    )

    # 4. 初始评估
    print("\n[4/6] Initial evaluation...")
    initial_train_mae = evaluate(model, train_loader, device, CONFIG['use_camera_condition'])
    initial_test_mae = evaluate(model, test_loader, device, CONFIG['use_camera_condition'])
    print(f"Initial Train MAE: {initial_train_mae:.2f} cm")
    print(f"Initial Test MAE:  {initial_test_mae:.2f} cm")

    # 5. 训练循环
    print("\n[5/6] Starting training...")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    if CONFIG['use_adel_prior']:
        print(f"ADEL Prior Constraint: ENABLED (lambda={CONFIG['prior_loss_weight']})")
    else:
        print(f"ADEL Prior Constraint: DISABLED")

    train_mae_history = []
    test_mae_history = []
    train_height_loss_history = []  # 高度损失
    train_prior_loss_history = []   # 先验损失
    best_test_mae = float('inf')
    best_epoch = 0

    for epoch in range(CONFIG['num_epochs']):
        # 训练阶段
        model.train()
        epoch_height_loss = 0.0  # 高度损失
        epoch_prior_loss = 0.0   # 先验损失
        epoch_total_loss = 0.0   # 总损失

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        for batch in pbar:
            features = batch['features'].to(device)  # (batch, 1024)
            targets = batch['target'].to(device)     # (batch, 1)
            cam_heights = batch['cam_height'].to(device)  # (batch, 1)

            # 前向传播
            optimizer.zero_grad()

            if CONFIG['use_camera_condition']:
                outputs = model(features, cam_heights.squeeze(-1))
            else:
                outputs = model(features)

            # 从 64D phytomer 参数计算预测高度
            internode_lengths = outputs[:, 0::4]  # (batch, 16)
            predicted_height = internode_lengths.sum(dim=1, keepdim=True)  # (batch, 1)

            # 高度监督损失 (Huber Loss)
            height_loss = criterion(predicted_height, targets)

            # ADEL 先验约束损失 (Synthetic-to-Real Prior Constraint)
            prior_loss = torch.tensor(0.0, device=device)
            if CONFIG['use_adel_prior'] and adel_prior is not None:
                adel_mean = adel_prior['mean']  # (64,)

                if CONFIG['prior_loss_type'] == 'l2':
                    # L2 距离: ||outputs - μ||^2
                    prior_loss = torch.mean((outputs - adel_mean) ** 2)
                elif CONFIG['prior_loss_type'] == 'mahalanobis':
                    # 马氏距离 (考虑参数间相关性)
                    diff = outputs - adel_mean  # (batch, 64)
                    inv_cov = adel_prior['inv_cov']  # (64, 64)
                    # 正确计算: (diff @ inv_cov) * diff, 然后 sum
                    mahal_dist = torch.sum((diff @ inv_cov) * diff, dim=1)
                    prior_loss = torch.mean(mahal_dist)

            # 总损失 = 高度损失 + λ * 先验损失
            total_loss = height_loss + CONFIG['prior_loss_weight'] * prior_loss

            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_height_loss += height_loss.item()
            epoch_prior_loss += prior_loss.item()
            epoch_total_loss += total_loss.item()

            pbar.set_postfix({
                'height_loss': f"{height_loss.item():.4f}",
                'prior_loss': f"{prior_loss.item():.4f}",
                'total': f"{total_loss.item():.4f}"
            })

        # 更新学习率
        scheduler.step()

        # 记录损失历史
        avg_height_loss = epoch_height_loss / len(train_loader)
        avg_prior_loss = epoch_prior_loss / len(train_loader)
        train_height_loss_history.append(avg_height_loss)
        train_prior_loss_history.append(avg_prior_loss)

        # 定期评估
        if (epoch + 1) % CONFIG['print_freq'] == 0 or epoch == 0:
            train_mae = evaluate(model, train_loader, device, CONFIG['use_camera_condition'])
            test_mae = evaluate(model, test_loader, device, CONFIG['use_camera_condition'])

            train_mae_history.append(train_mae)
            test_mae_history.append(test_mae)

            print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
            print(f"  Train MAE:    {train_mae:.2f} cm")
            print(f"  Test MAE:     {test_mae:.2f} cm")
            print(f"  Height Loss:  {avg_height_loss:.4f}")
            if CONFIG['use_adel_prior']:
                print(f"  Prior Loss:   {avg_prior_loss:.4f} (L2 dist to ADEL mean)")
            print(f"  LR:           {optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if test_mae < best_test_mae:
                best_test_mae = test_mae
                best_epoch = epoch + 1
                default_best_name = 'diffcorn_fusion_hand_prior_best.pth' if CONFIG['use_adel_prior'] else 'diffcorn_fusion_hand_best.pth'
                best_path = os.path.join(CONFIG['output_dir'], CONFIG['best_model_name'] or default_best_name)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'config': CONFIG,
                    'adel_prior': adel_prior if CONFIG['use_adel_prior'] else None,
                }, best_path)
                print(f"  *** Best model saved: {best_path} (MAE={test_mae:.2f} cm) ***")

    # 6. 保存最终模型
    print("\n[6/6] Saving final model...")
    final_filename = CONFIG['final_model_name'] or ('diffcorn_fusion_hand_prior_final.pth' if CONFIG['use_adel_prior'] else 'diffcorn_fusion_hand_final.pth')
    final_path = os.path.join(CONFIG['output_dir'], final_filename)
    torch.save({
        'epoch': CONFIG['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_mae': train_mae_history[-1] if train_mae_history else initial_train_mae,
        'test_mae': test_mae_history[-1] if test_mae_history else initial_test_mae,
        'best_test_mae': best_test_mae,
        'best_epoch': best_epoch,
        'config': CONFIG,
        'adel_prior': adel_prior if CONFIG['use_adel_prior'] else None,
    }, final_path)
    print(f"Final model saved: {final_path}")

    # 7. 绘制训练曲线
    print("\nPlotting training curves...")
    if CONFIG['use_adel_prior']:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax1, ax2, ax3 = axes
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax1, ax2 = axes

    # MAE 曲线
    epochs_eval = list(range(0, CONFIG['num_epochs'], CONFIG['print_freq']))
    if 0 not in epochs_eval:
        epochs_eval = [0] + epochs_eval

    ax1.plot(range(len(train_mae_history)), train_mae_history, 'o-', label='Train MAE', linewidth=2)
    ax1.plot(range(len(test_mae_history)), test_mae_history, 's-', label='Test MAE', linewidth=2)
    ax1.axhline(y=best_test_mae, color='g', linestyle='--', alpha=0.5, label=f'Best ({best_test_mae:.2f} cm)')
    ax1.set_xlabel('Evaluation Step')
    ax1.set_ylabel('MAE (cm)')
    ax1.set_title('Training Curves (Mixed-Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 预测 vs 真值散点图
    preds, targets, filenames = evaluate_detailed(model, test_loader, device, CONFIG['use_camera_condition'])

    ax2.scatter(targets, preds, alpha=0.6, s=30, edgecolors='black')
    ax2.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', label='Perfect prediction')

    # 线性拟合
    z = np.polyfit(targets, preds, 1)
    p = np.poly1d(z)
    ax2.plot(targets, p(targets), 'b-', label=f'Fit: y={z[0]:.2f}x+{z[1]:.1f}')

    ax2.set_xlabel('True Height (cm)')
    ax2.set_ylabel('Predicted Height (cm)')
    ax2.set_title(f'Test Set Predictions (MAE={test_mae_history[-1]:.2f} cm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ADEL 先验损失曲线
    if CONFIG['use_adel_prior']:
        ax3.plot(range(len(train_height_loss_history)), train_height_loss_history, 'o-', label='Height Loss', linewidth=2)
        ax3.plot(range(len(train_prior_loss_history)), train_prior_loss_history, 's-', label='Prior Loss (ADEL)', linewidth=2)
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss Value')
        ax3.set_title('Loss Decomposition')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

    plt.tight_layout()
    curve_path = os.path.join(CONFIG['output_dir'], CONFIG['curve_name'])
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved: {curve_path}")

    # 8. 生成评估报告
    print("\nGenerating evaluation report...")
    report_lines = [
        "=" * 70,
        "DiffCorn-Fusion Training Report - Hand BBOX",
        "=" * 70,
        "",
        "Configuration",
        "-" * 40,
        f"Bundle:           {CONFIG['bundle_path']}",
        f"Camera condition: {CONFIG['use_camera_condition']}",
        f"Cam H scale:      {CONFIG['cam_h_scale']}",
        f"ADEL Prior:       {CONFIG['use_adel_prior']}",
    ]

    if CONFIG['use_adel_prior']:
        report_lines.extend([
            f"  Prior path:     {CONFIG['adel_prior_path']}",
            f"  Prior loss:     {CONFIG['prior_loss_type']}",
            f"  Prior weight:   {CONFIG['prior_loss_weight']}",
        ])

    report_lines.extend([
        f"Batch size:       {CONFIG['batch_size']}",
        f"Epochs:           {CONFIG['num_epochs']}",
        f"Learning rate:    {CONFIG['learning_rate']}",
        f"Weight decay:     {CONFIG['weight_decay']}",
        f"Huber delta:      {CONFIG['delta']}",
        f"Seed:             {CONFIG['seed']}",
        "",
        "Training Results",
        "-" * 40,
        f"Initial Train MAE: {initial_train_mae:.2f} cm",
        f"Initial Test MAE:  {initial_test_mae:.2f} cm",
        f"Final Train MAE:   {train_mae_history[-1]:.2f} cm",
        f"Final Test MAE:    {test_mae_history[-1]:.2f} cm",
        f"Best Test MAE:     {best_test_mae:.2f} cm (Epoch {best_epoch})",
        "",
        "=" * 70,
    ])

    report_path = os.path.join(CONFIG['output_dir'], CONFIG['report_name'])
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"Report saved: {report_path}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Initial Test MAE:  {initial_test_mae:.2f} cm")
    print(f"Final Test MAE:    {test_mae_history[-1]:.2f} cm")
    print(f"Best Test MAE:     {best_test_mae:.2f} cm (Epoch {best_epoch})")
    print(f"\nModel saved: {final_path}")
    print(f"Best model: {best_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DiffCorn-Fusion on hand-annotated BBOX data.")
    parser.add_argument("--bundle-path", type=str, default=None)
    parser.add_argument("--adel-prior-path", type=str, default=None)
    parser.add_argument("--use-adel-prior", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--prior-loss-weight", type=float, default=None)
    parser.add_argument("--prior-loss-type", type=str, choices=["l2", "mahalanobis"], default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--best-model-name", type=str, default=None)
    parser.add_argument("--final-model-name", type=str, default=None)
    parser.add_argument("--report-name", type=str, default=None)
    parser.add_argument("--curve-name", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.bundle_path is not None:
        CONFIG['bundle_path'] = args.bundle_path
    if args.adel_prior_path is not None:
        CONFIG['adel_prior_path'] = args.adel_prior_path
    if args.use_adel_prior is not None:
        CONFIG['use_adel_prior'] = args.use_adel_prior.lower() == "true"
    if args.prior_loss_weight is not None:
        CONFIG['prior_loss_weight'] = args.prior_loss_weight
    if args.prior_loss_type is not None:
        CONFIG['prior_loss_type'] = args.prior_loss_type
    if args.output_dir is not None:
        CONFIG['output_dir'] = args.output_dir
    if args.best_model_name is not None:
        CONFIG['best_model_name'] = args.best_model_name
    if args.final_model_name is not None:
        CONFIG['final_model_name'] = args.final_model_name
    if args.report_name is not None:
        CONFIG['report_name'] = args.report_name
    if args.curve_name is not None:
        CONFIG['curve_name'] = args.curve_name
    if args.num_epochs is not None:
        CONFIG['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        CONFIG['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        CONFIG['learning_rate'] = args.learning_rate
    if args.seed is not None:
        CONFIG['seed'] = args.seed

    train_diffcorn_fusion()
