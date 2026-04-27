import math
import os

import kornia
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from multiview_detector.models.GaussianProbModel import GaussianLikelihoodEstimation
from multiview_detector.models.resnet import resnet18
from multiview_detector.utils.random_drop_frame import random_drop_frame
from multiview_detector.utils.random_drop_frame import random_drop_frame_with_priority
from multiview_detector.utils.random_drop_frame import random_drop_frame_with_priority_legacy


class TemporalEntropyModel(nn.Module):
    def __init__(self, tau_2, channel):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(tau_2 * channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2 * channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.conv_layers(x)


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AdaptiveTemporalFusionModule(nn.Module):
    def __init__(self, in_channels, num_cam, tau_1):
        super().__init__()
        total_in_channels = in_channels + num_cam * (tau_1 + 1)
        self.num_cam = num_cam
        self.tau_1 = tau_1
        self.conv1 = nn.Conv2d(total_in_channels, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(512, 1, kernel_size=3, padding=4, dilation=4, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        b, _, h, w = x.shape
        mask = mask.view(b, self.num_cam * (self.tau_1 + 1), 1, 1).expand(
            b, self.num_cam * (self.tau_1 + 1), h, w
        )
        x = torch.cat([x, mask], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)


class ChannelSemanticAwareJSCC(nn.Module):
    def __init__(
        self,
        in_channels,
        latent_channels,
        channel_type='rayleigh',
        eps=1e-6,
        csi_gain_scale=0.6,
        importance_gain_scale=1.0,
        low_snr_disable_csi_threshold=0.0,
        min_rate_scale=0.25,
    ):
        super().__init__()
        self.channel_type = channel_type
        self.eps = eps
        self.csi_gain_scale = csi_gain_scale
        self.importance_gain_scale = importance_gain_scale
        self.low_snr_disable_csi_threshold = low_snr_disable_csi_threshold
        self.min_rate_scale = min_rate_scale
        mid = max(in_channels // 2, 1)

        self.importance_head = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2 * latent_channels, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(2 * latent_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        )
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2 * latent_channels),
        )

    def apply_channel(self, z, snr_db):
        b = z.shape[0]
        snr_linear = torch.pow(10.0, snr_db / 10.0).view(b, 1, 1, 1)
        noise_std = torch.sqrt(1.0 / (2.0 * snr_linear + self.eps))
        if self.channel_type == 'awgn':
            return z + torch.randn_like(z) * noise_std

        zr, zi = torch.chunk(z, 2, dim=1)
        hr = torch.randn_like(zr) / math.sqrt(2.0)
        hi = torch.randn_like(zi) / math.sqrt(2.0)
        nr = torch.randn_like(zr) * noise_std
        ni = torch.randn_like(zi) * noise_std

        yr = hr * zr - hi * zi + nr
        yi = hr * zi + hi * zr + ni
        denom = hr.pow(2) + hi.pow(2) + self.eps
        eqr = (yr * hr + yi * hi) / denom
        eqi = (yi * hr - yr * hi) / denom
        return torch.cat([eqr, eqi], dim=1)

    def forward(self, x, snr_db, rate_scale=None):
        b = x.shape[0]
        importance = self.importance_head(x)
        z = self.encoder(x)

        snr_norm = (snr_db / 20.0).clamp(-1.5, 1.5)
        snr_gain = torch.sigmoid(self.snr_mlp(snr_norm)).view(b, -1, 1, 1)

        if self.low_snr_disable_csi_threshold is not None and self.low_snr_disable_csi_threshold > -999:
            csi_mask = (snr_db >= self.low_snr_disable_csi_threshold).float().view(b, 1, 1, 1)
        else:
            csi_mask = 1.0

        z = z * (1.0 + self.csi_gain_scale * snr_gain * csi_mask) * (1.0 + self.importance_gain_scale * importance)
        if rate_scale is None:
            rate_gate = torch.ones(b, 1, 1, 1, device=x.device, dtype=x.dtype)
        else:
            rate_gate = rate_scale.view(b, 1, 1, 1).clamp(self.min_rate_scale, 1.0)
        z = z * rate_gate

        power = z.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        z = z / torch.sqrt(power + self.eps)
        z_noisy = self.apply_channel(z, snr_db)
        x_hat = x + self.decoder(z_noisy)
        return x_hat, importance, rate_gate.view(b)


class CrossViewSemanticDenoising(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, bev_views):
        b, n, c, h, w = bev_views.shape
        global_query = bev_views.mean(dim=1)
        q = self.q_proj(global_query).view(b, 1, self.num_heads, self.head_dim, h, w)
        k = self.k_proj(bev_views.reshape(b * n, c, h, w)).view(b, n, self.num_heads, self.head_dim, h, w)
        v = self.v_proj(bev_views.reshape(b * n, c, h, w)).view(b, n, self.num_heads, self.head_dim, h, w)
        score = (q * k).sum(dim=3) / math.sqrt(self.head_dim)
        attn = torch.softmax(score, dim=1)
        fused = (attn.unsqueeze(3) * v).sum(dim=1).reshape(b, c, h, w)
        fused = self.out_proj(fused) + global_query
        return fused, attn.mean(dim=2)


class SNRGuidedCrossViewFusion(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn_fuser = CrossViewSemanticDenoising(channels, num_heads=num_heads)
        mid = max(channels // 4, 8)
        self.view_confidence = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1),
        )
        self.snr_gate = nn.Sequential(nn.Linear(1, channels), nn.Sigmoid())
        self.reliability_proj = nn.Sequential(
            nn.Linear(3, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, 1),
        )

    def forward(self, bev_views, snr_db=None, view_reliability=None):
        b, n, c, h, w = bev_views.shape
        fused_attn, attn_map = self.attn_fuser(bev_views)
        conf_logits = self.view_confidence(bev_views.reshape(b * n, c, h, w)).reshape(b, n, 1, h, w)
        if view_reliability is not None:
            rel_bias = self.reliability_proj(view_reliability.reshape(b * n, 3)).reshape(b, n, 1, 1, 1)
            conf_logits = conf_logits + rel_bias
        conf_weights = torch.softmax(conf_logits, dim=1)
        fused_conf = (conf_weights * bev_views).sum(dim=1)

        if snr_db is None:
            gate = fused_attn.new_full((b, c, 1, 1), 0.5)
        else:
            snr_norm = (snr_db / 20.0).clamp(-1.5, 1.5)
            gate = self.snr_gate(snr_norm).view(b, c, 1, 1)
        fused = gate * fused_attn + (1.0 - gate) * fused_conf
        return fused, attn_map, conf_weights


class EnhancedMapHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        hidden = 256
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.branch_d1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )
        self.branch_d2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.branch_d4 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.stem(x)
        b1 = self.branch_d1(x)
        b2 = self.branch_d2(x)
        b4 = self.branch_d4(x)
        return self.merge(torch.cat([b1, b2, b4], dim=1))


class RefinedCameraSelector(nn.Module):
    def __init__(self, feature_channels, hidden_dim=64, scale_min=0.6, scale_max=1.0):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        input_dim = feature_channels * 3 + 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim, feature_channels),
            nn.Sigmoid(),
        )

    def forward(self, current_stats, history_stats, uncertainty_stats, aux_stats):
        x = torch.cat([current_stats, history_stats, uncertainty_stats, aux_stats], dim=-1)
        hidden = self.mlp(x)
        score = self.score_head(hidden).squeeze(-1)
        scale = self.scale_min + (self.scale_max - self.scale_min) * self.scale_head(hidden)
        return score, scale


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.args = args
        self.num_cam = dataset.num_cam
        self.img_shape = dataset.img_shape
        self.reducedgrid_shape = dataset.reducedgrid_shape
        self.tau_2 = args.tau_2
        self.tau_1 = args.tau_1
        self.drop_prob = args.drop_prob
        self.method = getattr(args, 'method', 'baseline')
        self.disable_quantization = getattr(args, 'disable_quantization', False)
        self.enable_calibration_error = self.method == 'baseline'
        self.enable_map_visual_dump = self.method == 'baseline'
        self.refine_keep_cameras = getattr(args, 'refine_keep_cameras', -1)
        self.refine_keep_ratio = getattr(args, 'refine_keep_ratio', 1.0)
        self.refine_min_keep_cameras = getattr(args, 'refine_min_keep_cameras', 1)
        self.refine_score_mode = getattr(args, 'refine_score_mode', 'current')
        self.refine_score_noise_std = getattr(args, 'refine_score_noise_std', 0.0)
        self.refine_weighted_entropy = getattr(args, 'refine_weighted_entropy', False)
        self.refine_enable_token_drop = getattr(args, 'refine_enable_token_drop', False)
        self.refine_soft_weighting = getattr(args, 'refine_soft_weighting', False)
        self.refine_adaptive_keep_margin = getattr(args, 'refine_adaptive_keep_margin', 0.0)
        self.refine_scale_min = getattr(args, 'refine_scale_min', 0.6)
        self.refine_scale_max = getattr(args, 'refine_scale_max', 1.0)
        self.refine_apply_entropy_scale = getattr(args, 'refine_apply_entropy_scale', False)
        self.refine_channel_keep_ratio = getattr(args, 'refine_channel_keep_ratio', 1.0)
        self.refine_channel_min_keep = getattr(args, 'refine_channel_min_keep', 1)
        self.refine_channel_drop_floor = getattr(args, 'refine_channel_drop_floor', 0.0)

        self.jscc_channel_type = getattr(args, 'jscc_channel_type', 'rayleigh')
        self.snr_min_db = getattr(args, 'snr_min_db', 0.0)
        self.snr_max_db = getattr(args, 'snr_max_db', 20.0)
        self.test_snr_db = getattr(args, 'test_snr_db', 5.0)
        self.cross_view_heads = getattr(args, 'cross_view_heads', 4)
        self.jscc_csi_gain_scale = getattr(args, 'jscc_csi_gain_scale', 0.6)
        self.jscc_importance_gain_scale = getattr(args, 'jscc_importance_gain_scale', 1.0)
        self.jscc_low_snr_disable_csi_threshold = getattr(args, 'jscc_low_snr_disable_csi_threshold', 0.0)
        self.ablate_no_jscc = getattr(args, 'ablate_no_jscc', False)
        self.ablate_no_csi = getattr(args, 'ablate_no_csi', False)
        self.ablate_no_analog_channel = getattr(args, 'ablate_no_analog_channel', False)
        self.ablate_no_cross_view = getattr(args, 'ablate_no_cross_view', False)
        self.rate_aware_training = getattr(args, 'rate_aware_training', False)
        self.target_comm_kb = getattr(args, 'target_comm_kb', 28.5)
        self.min_keep_per_camera = getattr(args, 'min_keep_per_camera', 1)
        self.keep_latest_token = getattr(args, 'keep_latest_token', True)
        self.frame_dropout_noise_std = getattr(args, 'frame_dropout_noise_std', 0.05)
        self.lambda_consistency = getattr(args, 'lambda_consistency', 0.03)
        self.rate_view_dropout_prob = getattr(args, 'rate_view_dropout_prob', 0.2)

        self.intrinsic_matrices = dataset.base.intrinsic_matrices
        self.extrinsic_matrices = dataset.base.extrinsic_matrices
        self.worldgrid2worldcoord_mat = dataset.base.worldgrid2worldcoord_mat
        self.original_extrinsic_matrices = np.array(self.extrinsic_matrices, dtype=np.float64, copy=True)
        self.translation_error = 0.0
        self.rotation_error = 0.0
        self.error_camera = []
        self.epoch_thres = 10

        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(
            dataset.base.intrinsic_matrices,
            dataset.base.extrinsic_matrices,
            dataset.base.worldgrid2worldcoord_mat,
        )
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        self.img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        self.map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        self.proj_mats = [
            torch.from_numpy(self.map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ self.img_zoom_mat)
            for cam in range(self.num_cam)
        ]

        base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])
        split = 7
        self.base_pt1 = base[:split].to('cuda:1')
        self.base_pt2 = base[split:].to('cuda:0')
        self.channel = 8
        self.feature_extraction = nn.Conv2d(512, self.channel, 1).to('cuda:0')
        self.temporal_entropy_model = TemporalEntropyModel(self.tau_2, self.channel).to('cuda:0')
        self.temporal_fusion_module = AdaptiveTemporalFusionModule(
            in_channels=self.channel * self.num_cam * (self.tau_1 + 1) + 2,
            num_cam=self.num_cam,
            tau_1=self.tau_1,
        ).to('cuda:0')
        self.refined_selector = RefinedCameraSelector(
            self.channel,
            scale_min=self.refine_scale_min,
            scale_max=self.refine_scale_max,
        ).to('cuda:0')

        self.proposed_tx_channels = (self.tau_1 + 1) * self.channel
        if self.method == 'proposed_jscc':
            proposed_latent_channels = getattr(args, 'jscc_latent_channels', -1)
            if proposed_latent_channels <= 0:
                proposed_latent_channels = self.proposed_tx_channels
            self.proposed_jscc = ChannelSemanticAwareJSCC(
                in_channels=self.proposed_tx_channels,
                latent_channels=proposed_latent_channels,
                channel_type=self.jscc_channel_type,
                csi_gain_scale=self.jscc_csi_gain_scale,
                importance_gain_scale=self.jscc_importance_gain_scale,
                low_snr_disable_csi_threshold=self.jscc_low_snr_disable_csi_threshold,
                min_rate_scale=0.25,
            ).to('cuda:0')
            self.proposed_cross_view = SNRGuidedCrossViewFusion(
                channels=self.proposed_tx_channels,
                num_heads=self.cross_view_heads,
            ).to('cuda:0')
            self.proposed_map_head = EnhancedMapHead(self.proposed_tx_channels + 2).to('cuda:0')

        self._load_checkpoint(self.args.model_path)

        for param in self.base_pt1.parameters():
            param.requires_grad = False
        for param in self.base_pt2.parameters():
            param.requires_grad = False
        for param in self.feature_extraction.parameters():
            param.requires_grad = False

    def _load_checkpoint(self, pretrained_model_path):
        if not pretrained_model_path or (not os.path.exists(pretrained_model_path)):
            raise FileNotFoundError(f"model_path not found: {pretrained_model_path}")

        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        model_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

        has_full_detector_weights = any(
            k.startswith('temporal_entropy_model')
            or k.startswith('temporal_fusion_module')
            or k.startswith('proposed_jscc')
            or k.startswith('proposed_cross_view')
            or k.startswith('proposed_map_head')
            for k in model_dict
        )

        if self.method == 'proposed_jscc':
            missing_keys, unexpected_keys = self.load_state_dict(model_dict, strict=False)
            print(f"Loaded checkpoint from {pretrained_model_path}")
            print(f"Checkpoint load summary: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
            return

        if has_full_detector_weights:
            missing_keys, unexpected_keys = self.load_state_dict(model_dict, strict=False)
            print(f"Loaded full baseline/refined checkpoint from {pretrained_model_path}")
            print(f"Checkpoint load summary: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
            return

        base_pt1_dict = {k[9:]: v for k, v in model_dict.items() if k[:8] == 'base_pt1'}
        base_pt2_dict = {k[9:]: v for k, v in model_dict.items() if k[:8] == 'base_pt2'}
        feature_extraction_dict = {k[19:]: v for k, v in model_dict.items() if k[:18] == 'feature_extraction'}
        self.base_pt1.load_state_dict(base_pt1_dict)
        self.base_pt2.load_state_dict(base_pt2_dict)
        self.feature_extraction.load_state_dict(feature_extraction_dict)
        print(f"Loaded baseline backbone weights from {pretrained_model_path}")

    def process_features_with_temporal_fusion(self, cam_feature, coord_map, cam_idx=0, is_training=True):
        b, c, h, w = cam_feature.shape
        expected_channels = self.channel * (self.tau_1 + 1)
        if c != expected_channels:
            raise ValueError(f"cam_feature channel mismatch: expected {expected_channels}, got {c}")
        total_channels = self.channel * self.num_cam * (self.tau_1 + 1)
        expanded_cam_feature = torch.zeros(
            b, total_channels, h, w, dtype=cam_feature.dtype, device=cam_feature.device
        )
        start = cam_idx * expected_channels
        end = start + expected_channels
        expanded_cam_feature[:, start:end, :, :] = cam_feature
        world_features_with_coord = torch.cat(
            [expanded_cam_feature, coord_map.repeat([b, 1, 1, 1]).to(cam_feature.device)],
            dim=1,
        )
        mask = torch.ones(b, self.num_cam * (self.tau_1 + 1), dtype=torch.float32, device=cam_feature.device)
        return self.temporal_fusion_module(world_features_with_coord, mask)

    def save_map_result_images(self, world_features, coord_map, save_dir):
        b, _, _, _ = world_features.shape
        os.makedirs(save_dir, exist_ok=True)
        for cam_num in range(self.num_cam):
            start = cam_num * self.channel * (self.tau_1 + 1)
            end = start + self.channel * (self.tau_1 + 1)
            cam_feature = world_features[:, start:end, :, :]
            map_result = self.process_features_with_temporal_fusion(
                cam_feature, coord_map, cam_idx=cam_num, is_training=self.training
            )
            for bi in range(b):
                result_image = map_result[bi, 0, :, :].detach().cpu().numpy()
                image_path = os.path.join(save_dir, f"batch_{bi}_camera_{cam_num}_map_result.png")
                plt.imsave(image_path, result_image, cmap='gray')

    def feature_extraction_step(self, imgs):
        b, n, c, h, w = imgs.shape
        imgs = torch.reshape(imgs, (b * n, c, h, w))
        img_feature = self.base_pt1(imgs.to('cuda:1'))
        img_feature = self.base_pt2(img_feature.to('cuda:0'))
        img_feature = self.feature_extraction(img_feature)
        if not self.disable_quantization:
            img_feature = torch.round(img_feature)
        _, c, h, w = img_feature.shape
        return torch.reshape(img_feature, (b, n, c, h, w))

    def sample_snr_db(self, batch_size, device):
        if self.training:
            return torch.empty(batch_size, 1, device=device).uniform_(self.snr_min_db, self.snr_max_db)
        return torch.full((batch_size, 1), self.test_snr_db, device=device)

    def sample_rate_scale(self, snr_db):
        snr_norm = (snr_db / 20.0).clamp(-1.5, 1.5)
        rate = 0.55 + 0.45 * torch.sigmoid(2.0 * snr_norm)
        return rate.clamp(0.25, 1.0)

    def _get_current_epoch(self):
        if not os.path.exists('epoch.log'):
            return 0
        try:
            with open('epoch.log', 'r') as f:
                return int(f.read().strip())
        except Exception:
            return 0

    def _maybe_apply_calibration_error(self, epoch):
        self.extrinsic_matrices = np.array(self.original_extrinsic_matrices, dtype=np.float64, copy=True)
        if (not self.enable_calibration_error) or self.training or epoch <= self.epoch_thres:
            return

        print(f"Epoch: {epoch}, Reading CSV for test parameters...")
        csv_path = os.path.join('temp', 'Calibration', 'calibration_test_rotation_error.csv')
        if not os.path.exists(csv_path):
            print(f"Calibration CSV not found: {csv_path}, skip error injection.")
            self.error_camera = []
            self.translation_error = 0.0
            self.rotation_error = 0.0
            return

        csv_data = pd.read_csv(csv_path)
        if "Epoch" not in csv_data.columns or csv_data[csv_data["Epoch"] == epoch].empty:
            print(f"No calibration row for epoch {epoch}, skip error injection.")
            self.error_camera = []
            self.translation_error = 0.0
            self.rotation_error = 0.0
            return

        test_params = csv_data[csv_data["Epoch"] == epoch].iloc[0]
        self.translation_error = test_params['Translation Error']
        self.rotation_error = test_params['Rotation Error']
        self.error_camera = test_params['error_camera']
        print(f"Loaded params - Translation Error: {self.translation_error}, Rotation Error: {self.rotation_error}, Error Camera: {self.error_camera}")

        if isinstance(self.error_camera, str):
            self.error_camera = [int(cam.strip()) for cam in self.error_camera.split(',') if cam.strip().isdigit()]
        elif isinstance(self.error_camera, (int, float)):
            self.error_camera = [int(self.error_camera)]
        elif not isinstance(self.error_camera, list):
            self.error_camera = []
        print(f"Error Camera List: {self.error_camera}")

        for cam in self.error_camera:
            cam = int(cam)
            if cam >= len(self.extrinsic_matrices):
                raise IndexError(f"Camera index {cam} is out of bounds for extrinsic_matrices.")
            r = self.extrinsic_matrices[cam][:, :3]
            t = self.extrinsic_matrices[cam][:, 3]
            rotation_perturbation = np.random.randn(3, 3) * self.rotation_error
            translation_perturbation = np.random.randn(3) * self.translation_error * 100
            self.extrinsic_matrices[cam][:, :3] = r * (1 + rotation_perturbation)
            self.extrinsic_matrices[cam][:, 3] = t * (1 + translation_perturbation)
        print(f"Applied translation error {self.translation_error} and rotation error {self.rotation_error} to cameras {self.error_camera}")

    def _update_projection_mats_if_needed(self, epoch):
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(
            self.intrinsic_matrices,
            self.extrinsic_matrices,
            self.worldgrid2worldcoord_mat,
        )
        self.proj_mats = [
            torch.from_numpy(self.map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ self.img_zoom_mat)
            for cam in range(self.num_cam)
        ]
        if self.enable_calibration_error and (not self.training) and epoch > self.epoch_thres:
            print("Updated projection matrices for testing phase.")

    def _resolve_refine_keep_count(self):
        if self.refine_keep_cameras > 0:
            keep_count = min(self.num_cam, self.refine_keep_cameras)
        elif self.refine_keep_ratio < 1.0:
            keep_count = int(math.ceil(self.num_cam * self.refine_keep_ratio))
        else:
            keep_count = self.num_cam
        return max(1, min(self.num_cam, max(self.refine_min_keep_cameras, keep_count)))

    def _pool_refined_stats(self, current_features, history_features, scales_hat):
        current_stats = current_features.abs().mean(dim=(3, 4))
        uncertainty_stats = scales_hat.abs().mean(dim=(3, 4))

        if self.tau_2 > 0:
            b, n, tc, h, w = history_features.shape
            expected_tc = self.tau_2 * self.channel
            if tc != expected_tc:
                raise ValueError(f"history_features channel mismatch: expected {expected_tc}, got {tc}")
            history_reshaped = history_features.view(b, n, self.tau_2, self.channel, h, w)
            history_stats = history_reshaped.abs().mean(dim=(2, 4, 5))
        else:
            history_stats = torch.zeros_like(current_stats)

        current_scalar = current_stats.mean(dim=2, keepdim=True)
        history_scalar = history_stats.mean(dim=2, keepdim=True)
        uncertainty_scalar = uncertainty_stats.mean(dim=2, keepdim=True)
        aux_stats = torch.cat([current_scalar - history_scalar, uncertainty_scalar], dim=2)
        return current_stats, history_stats, uncertainty_stats, aux_stats

    def _build_refined_camera_mask(self, current_features, history_features, scales_hat):
        b, n = current_features.shape[:2]
        keep_count = self._resolve_refine_keep_count()
        current_stats, history_stats, uncertainty_stats, aux_stats = self._pool_refined_stats(
            current_features,
            history_features,
            scales_hat,
        )
        score, scale = self.refined_selector(
            current_stats,
            history_stats,
            uncertainty_stats,
            aux_stats,
        )
        current_scalar = current_stats.mean(dim=2)
        history_scalar = history_stats.mean(dim=2)
        uncertainty_scalar = uncertainty_stats.mean(dim=2)

        if self.refine_score_mode == 'current':
            score = score + current_scalar
        elif self.refine_score_mode == 'current_temporal':
            score = score + current_scalar + 0.35 * history_scalar
        else:
            score = score + current_scalar + 0.35 * history_scalar - 0.15 * uncertainty_scalar

        if self.training and self.refine_score_noise_std > 0:
            score = score + torch.randn_like(score) * self.refine_score_noise_std

        if keep_count >= n:
            mask = torch.ones(b, n, device=current_features.device, dtype=current_features.dtype)
            return mask, score, scale

        extra_slot = 1 if self.refine_adaptive_keep_margin > 0 and keep_count < n else 0
        top_count = min(n, keep_count + extra_slot)
        top_scores, top_idx = torch.topk(score, k=top_count, dim=1, largest=True)
        mask = torch.zeros(b, n, device=current_features.device, dtype=current_features.dtype)
        mask.scatter_(1, top_idx[:, :keep_count], 1.0)
        if extra_slot:
            score_gap = top_scores[:, keep_count - 1] - top_scores[:, keep_count]
            rescue_rows = score_gap < self.refine_adaptive_keep_margin
            if rescue_rows.any():
                rescue_batch_idx = rescue_rows.nonzero(as_tuple=False).squeeze(1)
                rescue_cam_idx = top_idx[rescue_rows, keep_count]
                mask[rescue_batch_idx, rescue_cam_idx] = 1.0
        return mask, score, scale

    def _build_refined_channel_mask(self, camera_scale):
        if self.refine_channel_keep_ratio >= 1.0:
            return torch.ones_like(camera_scale)

        keep_channels = int(math.ceil(self.channel * self.refine_channel_keep_ratio))
        keep_channels = max(self.refine_channel_min_keep, keep_channels)
        keep_channels = min(self.channel, keep_channels)
        if keep_channels >= self.channel:
            return torch.ones_like(camera_scale)

        top_idx = torch.topk(camera_scale, k=keep_channels, dim=2, largest=True).indices
        channel_mask = torch.zeros_like(camera_scale)
        channel_mask.scatter_(2, top_idx, 1.0)
        if self.refine_channel_drop_floor > 0:
            channel_mask = channel_mask + (1.0 - channel_mask) * self.refine_channel_drop_floor
        return channel_mask

    def forward_proposed(self, imgs_list):
        b, t, n, c, h, w = imgs_list.shape
        assert n == self.num_cam
        tau = max(self.tau_1, self.tau_2)

        imgs_list_feature = []
        for i in range(tau + 1):
            imgs_feature = self.feature_extraction_step(imgs_list[:, i])
            imgs_list_feature.append(imgs_feature.unsqueeze(dim=1))
        imgs_list_feature = torch.cat(imgs_list_feature, dim=1)

        to_be_transmitted_feature = imgs_list_feature[:, self.tau_2]
        conditional_features = imgs_list_feature[:, :self.tau_2]
        conditional_features = torch.swapaxes(conditional_features, 1, 2)
        conditional_features = torch.reshape(conditional_features, (b, n, self.tau_2 * self.channel, 90, 160))
        conditional_features = torch.reshape(conditional_features, (b * n, self.tau_2 * self.channel, 90, 160))

        gaussian_params = self.temporal_entropy_model(conditional_features)
        gaussian_params = torch.reshape(gaussian_params, (b, n, 2 * self.channel, 90, 160))
        scales_hat, means_hat = gaussian_params.chunk(2, dim=2)
        feature_likelihoods = GaussianLikelihoodEstimation(to_be_transmitted_feature, scales_hat, means=means_hat)
        entropy_bits_loss = (torch.log(feature_likelihoods).sum() / (-math.log(2)))

        feature4prediction = imgs_list_feature[:, -(self.tau_1 + 1):]
        feature4prediction = torch.swapaxes(feature4prediction, 1, 2)
        feature4prediction = torch.reshape(feature4prediction, (b, n, self.proposed_tx_channels, 90, 160))
        feature4prediction = torch.reshape(feature4prediction, (b * n, self.proposed_tx_channels, 90, 160))
        feature4prediction = F.interpolate(feature4prediction, size=self.upsample_shape, mode='bilinear')
        feature4prediction = torch.reshape(feature4prediction, (b, n, self.proposed_tx_channels, 270, 480))

        flat_features = feature4prediction.reshape(b, n * self.proposed_tx_channels, 270, 480)
        if self.rate_aware_training:
            flat_features, tx_mask = random_drop_frame_with_priority(
                flat_features,
                num_cam=self.num_cam,
                tau_1=0,
                channel=self.proposed_tx_channels,
                target_dropout_rate=self.drop_prob,
                is_training=self.training,
                min_keep_per_camera=self.min_keep_per_camera,
                keep_latest_token=self.keep_latest_token,
                score_noise_std=self.frame_dropout_noise_std,
            )
        else:
            tx_mask = torch.ones(b, self.num_cam, dtype=flat_features.dtype, device=flat_features.device)
        feature4prediction = flat_features.view(b, n, self.proposed_tx_channels, 270, 480)

        snr_db = self.sample_snr_db(b, feature4prediction.device)
        if self.ablate_no_csi:
            snr_db = torch.full_like(snr_db, self.test_snr_db)
        global_rate_scale = self.sample_rate_scale(snr_db) if self.rate_aware_training else torch.ones(b, 1, device=snr_db.device)

        bev_views = []
        view_rate_scales = []
        view_snr_norm = (snr_db / 20.0).clamp(-1.5, 1.5)
        for cam in range(self.num_cam):
            cam_keep = tx_mask[:, cam:cam + 1]
            cam_feature = feature4prediction[:, cam] * cam_keep.view(b, 1, 1, 1)

            if self.ablate_no_jscc:
                tx_feat = cam_feature
                rate_used = cam_keep.view(b)
            else:
                if self.ablate_no_analog_channel:
                    tx_feat, _, rate_used = self.proposed_jscc(cam_feature, torch.full_like(snr_db, 100.0), rate_scale=global_rate_scale)
                else:
                    tx_feat, _, rate_used = self.proposed_jscc(cam_feature, snr_db, rate_scale=global_rate_scale)
                rate_used = rate_used * cam_keep.view(b)

            proj_mat = self.proj_mats[cam].repeat([b, 1, 1]).float().to('cuda:0')
            bev_views.append(kornia.geometry.transform.warp_perspective(tx_feat.to('cuda:0'), proj_mat, self.reducedgrid_shape))
            view_rate_scales.append(rate_used)

        bev_views = torch.stack(bev_views, dim=1)
        view_rate_tensor = torch.stack(view_rate_scales, dim=1)
        view_reliability = torch.stack(
            [view_snr_norm.repeat(1, self.num_cam), tx_mask, view_rate_tensor],
            dim=-1,
        )

        if self.ablate_no_cross_view:
            fused_bev = bev_views.mean(dim=1)
        else:
            fused_bev, _, _ = self.proposed_cross_view(bev_views, snr_db, view_reliability=view_reliability)

        coord = self.coord_map.repeat([b, 1, 1, 1]).to('cuda:0')
        map_result = self.proposed_map_head(torch.cat([fused_bev, coord], dim=1))

        entropy_kb = entropy_bits_loss / 8 / 1024
        keep_ratio = tx_mask.mean()
        rate_ratio = view_rate_tensor.mean()
        tx_kb_proxy = self.target_comm_kb * keep_ratio * rate_ratio

        if self.training and self.rate_aware_training and self.rate_view_dropout_prob > 0:
            random_drop = (torch.rand(b, self.num_cam, device=bev_views.device) > self.rate_view_dropout_prob).float()
            random_drop = torch.maximum(random_drop, tx_mask)
            random_drop = random_drop / (random_drop.sum(dim=1, keepdim=True) + 1e-6)
            weak_fused = (bev_views * random_drop.view(b, self.num_cam, 1, 1, 1)).sum(dim=1)
            consistency_loss = F.mse_loss(weak_fused, fused_bev.detach())
        else:
            consistency_loss = map_result.new_tensor(0.0)

        bits_loss = entropy_kb + tx_kb_proxy + self.lambda_consistency * consistency_loss

        if not self.training:
            map_results = []
            for cam_num in range(self.num_cam):
                map_result_single = self.proposed_map_head(torch.cat([bev_views[:, cam_num], coord], dim=1))
                map_results.append(map_result_single)
            map_results = torch.cat(map_results, dim=1)
        else:
            map_results = torch.zeros(b, self.num_cam, map_result.shape[-2], map_result.shape[-1], device=map_result.device)

        return map_result, bits_loss, map_results

    def forward_baseline(self, imgs_list, use_priority_drop=True):
        b, t, n, c, h, w = imgs_list.shape
        assert n == self.num_cam
        tau = max(self.tau_1, self.tau_2)

        imgs_list_feature = []
        for i in range(tau + 1):
            imgs_feature = self.feature_extraction_step(imgs_list[:, i])
            imgs_list_feature.append(imgs_feature.unsqueeze(dim=1))
        imgs_list_feature = torch.cat(imgs_list_feature, dim=1)
        assert t == imgs_list_feature.size()[1]

        to_be_transmitted_feature = imgs_list_feature[:, self.tau_2]
        conditional_features = imgs_list_feature[:, :self.tau_2]
        conditional_features = torch.swapaxes(conditional_features, 1, 2)
        conditional_features = torch.reshape(conditional_features, (b, n, self.tau_2 * self.channel, 90, 160))
        conditional_features_flat = torch.reshape(conditional_features, (b * n, self.tau_2 * self.channel, 90, 160))

        gaussian_params = self.temporal_entropy_model(conditional_features_flat)
        gaussian_params = torch.reshape(gaussian_params, (b, n, 2 * self.channel, 90, 160))
        scales_hat, means_hat = gaussian_params.chunk(2, dim=2)
        if self.method == 'baseline_refined':
            camera_keep_mask, camera_score, camera_scale = self._build_refined_camera_mask(
                to_be_transmitted_feature,
                conditional_features,
                scales_hat,
            )
            channel_keep_mask = self._build_refined_channel_mask(camera_scale)
        else:
            camera_keep_mask = torch.ones(b, n, device=to_be_transmitted_feature.device, dtype=to_be_transmitted_feature.dtype)
            camera_score = camera_keep_mask
            camera_scale = torch.ones(b, n, self.channel, device=to_be_transmitted_feature.device, dtype=to_be_transmitted_feature.dtype)
            channel_keep_mask = camera_scale

        likelihood_input = to_be_transmitted_feature
        if self.method == 'baseline_refined':
            likelihood_input = likelihood_input * camera_keep_mask.view(b, n, 1, 1, 1)
            likelihood_input = likelihood_input * channel_keep_mask.view(b, n, self.channel, 1, 1)
            if self.refine_apply_entropy_scale:
                likelihood_input = likelihood_input * camera_scale.view(b, n, self.channel, 1, 1)
        feature_likelihoods = GaussianLikelihoodEstimation(likelihood_input, scales_hat, means=means_hat)

        if self.method == 'baseline_refined' and self.refine_weighted_entropy:
            likelihood_mask = camera_keep_mask.view(b, n, 1, 1, 1) * channel_keep_mask.view(b, n, self.channel, 1, 1)
            bits_loss = (torch.log(feature_likelihoods) * likelihood_mask).sum() / (-math.log(2))
        else:
            bits_loss = (torch.log(feature_likelihoods).sum() / (-math.log(2)))

        feature4prediction = imgs_list_feature[:, -(self.tau_1 + 1):]
        feature4prediction = torch.swapaxes(feature4prediction, 1, 2)
        if self.method == 'baseline_refined':
            feature4prediction = feature4prediction * camera_keep_mask.view(b, n, 1, 1, 1, 1)
            repeated_channel_mask = channel_keep_mask.unsqueeze(2).repeat(1, 1, self.tau_1 + 1, 1).view(
                b, n, self.tau_1 + 1, self.channel, 1, 1
            )
            feature4prediction = feature4prediction * repeated_channel_mask
            repeated_scale = camera_scale.repeat_interleave(self.tau_1 + 1, dim=2).view(b, n, self.tau_1 + 1, self.channel, 1, 1)
            feature4prediction = feature4prediction * repeated_scale
        feature4prediction = torch.reshape(feature4prediction, (b, n, (self.tau_1 + 1) * self.channel, 90, 160))
        feature4prediction = torch.reshape(feature4prediction, (b * n, (self.tau_1 + 1) * self.channel, 90, 160))
        feature4prediction = F.interpolate(feature4prediction, size=self.upsample_shape, mode='bilinear')
        feature4prediction = torch.reshape(feature4prediction, (b, n, (self.tau_1 + 1) * self.channel, 270, 480))
        if self.method == 'baseline_refined' and self.refine_soft_weighting:
            masked_score = camera_score.masked_fill(camera_keep_mask < 0.5, float('-inf'))
            camera_weights = torch.softmax(masked_score, dim=1) * camera_keep_mask
            camera_weights = camera_weights / (camera_weights.sum(dim=1, keepdim=True) + 1e-6)
            feature4prediction = feature4prediction * camera_weights.view(b, n, 1, 1, 1)

        epoch = self._get_current_epoch()
        self._maybe_apply_calibration_error(epoch)
        self._update_projection_mats_if_needed(epoch)

        world_features = []
        for cam in range(self.num_cam):
            proj_mat = self.proj_mats[cam].repeat([b, 1, 1]).float().to('cuda:0')
            if self.enable_calibration_error and (not self.training) and epoch > self.epoch_thres:
                print(f"Updated projection matrix for camera {cam}.")
                print(f"Projection Matrix: {proj_mat}")
            world_feature = kornia.geometry.transform.warp_perspective(
                feature4prediction[:, cam].to('cuda:0'),
                proj_mat,
                self.reducedgrid_shape,
            )
            world_features.append(world_feature.to('cuda:0'))
        world_features = torch.cat(world_features, dim=1)
        camera_token_mask = camera_keep_mask.unsqueeze(-1).repeat(1, 1, self.tau_1 + 1).reshape(b, -1)

        if self.enable_map_visual_dump:
            self.save_map_result_images(world_features, self.coord_map, os.path.join('temp', 'map_res'))

        if self.method == 'baseline':
            world_features, mask = random_drop_frame_with_priority_legacy(
                world_features, self.num_cam, self.tau_1, self.channel, self.drop_prob, is_training=self.training
            )
        elif self.method == 'baseline_refined' and not self.refine_enable_token_drop:
            mask = camera_token_mask
            expanded_mask = mask.unsqueeze(2).repeat(1, 1, self.channel).view(b, world_features.shape[1], 1, 1)
            world_features = world_features * expanded_mask
        elif use_priority_drop:
            world_features, mask = random_drop_frame_with_priority(
                world_features, self.num_cam, self.tau_1, self.channel, self.drop_prob, is_training=self.training
            )
            mask = mask * camera_token_mask
            expanded_mask = mask.unsqueeze(2).repeat(1, 1, self.channel).view(b, world_features.shape[1], 1, 1)
            world_features = world_features * expanded_mask
        else:
            world_features, mask = random_drop_frame(
                world_features, self.num_cam, self.tau_1, self.channel, self.drop_prob
            )
            mask = mask * camera_token_mask
            expanded_mask = mask.unsqueeze(2).repeat(1, 1, self.channel).view(b, world_features.shape[1], 1, 1)
            world_features = world_features * expanded_mask

        world_features = torch.cat([world_features, self.coord_map.repeat([b, 1, 1, 1]).to('cuda:0')], dim=1)
        map_result = self.temporal_fusion_module(world_features, mask)
        bits_loss = bits_loss / 8 / 1024

        if not self.training:
            map_results = []
            for cam_num in range(self.num_cam):
                start = cam_num * self.channel * (self.tau_1 + 1)
                end = start + self.channel * (self.tau_1 + 1)
                cam_feature = world_features[:, start:end, :, :]
                map_result_single = self.process_features_with_temporal_fusion(
                    cam_feature,
                    self.coord_map,
                    cam_idx=cam_num,
                    is_training=self.training,
                )
                map_results.append(map_result_single)
            map_results = torch.cat(map_results, dim=1)
        else:
            map_results = torch.zeros(b, self.num_cam, h, w, device=world_features.device)

        return map_result, bits_loss, map_results

    def forward(self, imgs_list, visualize=False):
        if self.method == 'proposed_jscc':
            return self.forward_proposed(imgs_list)
        if self.method == 'baseline_refined':
            return self.forward_baseline(imgs_list, use_priority_drop=True)
        return self.forward_baseline(imgs_list, use_priority_drop=True)

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)
            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        h, w, _ = img_size
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_x = torch.from_numpy(grid_x / (w - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (h - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, h, w])
            ret = torch.cat([ret, rr], dim=1)
        return ret
