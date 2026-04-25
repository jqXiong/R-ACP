import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

from multiview_detector.models.resnet import resnet18
from multiview_detector.models.GaussianProbModel import GaussianLikelihoodEstimation
from multiview_detector.utils.random_drop_frame import random_drop_frame
from multiview_detector.utils.random_drop_frame import random_drop_frame_with_priority
from multiview_detector.models.Priority_network import LightweightPriorityNetwork

import cv2

import matplotlib.pyplot as plt
import math
import copy
import pandas as pd



class TemporalEntropyModel(nn.Module):
    def __init__(self, tau_2, channel):
        super(TemporalEntropyModel, self).__init__()
        self.tau_2 = tau_2
        self.channel = channel

        self.conv_layers = nn.Sequential(
            nn.Conv2d(tau_2 * channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2 * channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # x: (B * N, tau_2 * channel, H, W)
        x = self.conv_layers(x)  # (B * N, 2 * channel, H, W)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class AdaptiveTemporalFusionModule(nn.Module):
    def __init__(self, in_channels, num_cam, tau_1):
        super(AdaptiveTemporalFusionModule, self).__init__()
        self.num_cam = num_cam
        self.tau_1 = tau_1
        # 修改卷积层输入通道数为 feature 通道数 + mask 通道数
        total_in_channels = in_channels + num_cam * (tau_1 + 1)  # features + mask 通道数
        self.conv1 = nn.Conv2d(total_in_channels, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(512, 1, kernel_size=3, padding=4, dilation=4, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        B, C, H, W = x.shape
        
        # mask 形状为 [B, num_cam * (tau_1 + 1)]，扩展为 [B, num_cam * (tau_1 + 1), H, W]
        mask = mask.view(B, self.num_cam * (self.tau_1 + 1), 1, 1).expand(B, self.num_cam * (self.tau_1 + 1), H, W)
        
        # 合并 mask 和 features
        x = torch.cat([x, mask], dim=1)  # 合并后的 x 应该是 [B, C + num_cam * (tau_1 + 1), H, W]

        # 经过卷积层处理
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class ChannelSemanticAwareJSCC(nn.Module):
    def __init__(self, in_channels, latent_channels, channel_type='rayleigh', eps=1e-6,
                 csi_gain_scale=0.6, importance_gain_scale=1.0, low_snr_disable_csi_threshold=0.0,
                 min_rate_scale=0.25):
        super(ChannelSemanticAwareJSCC, self).__init__()
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
            nn.Sigmoid()
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2 * latent_channels, kernel_size=1)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(2 * latent_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )

        self.snr_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2 * latent_channels)
        )

    def apply_channel(self, z, snr_db):
        B = z.shape[0]
        snr_linear = torch.pow(10.0, snr_db / 10.0).view(B, 1, 1, 1)
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
        B = x.shape[0]
        importance = self.importance_head(x)
        z = self.encoder(x)

        snr_norm = (snr_db / 20.0).clamp(-1.5, 1.5)
        snr_gain = torch.sigmoid(self.snr_mlp(snr_norm)).view(B, -1, 1, 1)

        if self.low_snr_disable_csi_threshold is not None and self.low_snr_disable_csi_threshold > -999:
            csi_mask = (snr_db >= self.low_snr_disable_csi_threshold).float().view(B, 1, 1, 1)
        else:
            csi_mask = 1.0

        z = z * (1.0 + self.csi_gain_scale * snr_gain * csi_mask) * (1.0 + self.importance_gain_scale * importance)

        if rate_scale is None:
            rate_gate = torch.ones(B, 1, 1, 1, device=x.device, dtype=x.dtype)
        else:
            rate_gate = rate_scale.view(B, 1, 1, 1).clamp(self.min_rate_scale, 1.0)
        z = z * rate_gate

        power = z.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        z = z / torch.sqrt(power + self.eps)

        z_noisy = self.apply_channel(z, snr_db)
        x_hat = x + self.decoder(z_noisy)
        return x_hat, importance, rate_gate.view(B)


class CrossViewSemanticDenoising(nn.Module):
    def __init__(self, channels, num_heads=4):
        super(CrossViewSemanticDenoising, self).__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, bev_views):
        B, N, C, H, W = bev_views.shape
        global_query = bev_views.mean(dim=1)

        q = self.q_proj(global_query).view(B, 1, self.num_heads, self.head_dim, H, W)
        k = self.k_proj(bev_views.reshape(B * N, C, H, W)).view(B, N, self.num_heads, self.head_dim, H, W)
        v = self.v_proj(bev_views.reshape(B * N, C, H, W)).view(B, N, self.num_heads, self.head_dim, H, W)

        score = (q * k).sum(dim=3) / math.sqrt(self.head_dim)
        attn = torch.softmax(score, dim=1)

        fused = (attn.unsqueeze(3) * v).sum(dim=1).reshape(B, C, H, W)
        fused = self.out_proj(fused) + global_query
        return fused, attn.mean(dim=2)


class SNRGuidedCrossViewFusion(nn.Module):
    def __init__(self, channels, num_heads=4):
        super(SNRGuidedCrossViewFusion, self).__init__()
        self.attn_fuser = CrossViewSemanticDenoising(channels, num_heads=num_heads)
        mid = max(channels // 4, 8)
        self.view_confidence = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1)
        )
        self.snr_gate = nn.Sequential(
            nn.Linear(1, channels),
            nn.Sigmoid()
        )
        self.reliability_proj = nn.Sequential(
            nn.Linear(3, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, 1)
        )

    def forward(self, bev_views, snr_db=None, view_reliability=None):
        B, N, C, H, W = bev_views.shape
        fused_attn, attn_map = self.attn_fuser(bev_views)

        conf_logits = self.view_confidence(bev_views.reshape(B * N, C, H, W)).reshape(B, N, 1, H, W)
        if view_reliability is not None:
            rel_bias = self.reliability_proj(view_reliability.reshape(B * N, 3)).reshape(B, N, 1, 1, 1)
            conf_logits = conf_logits + rel_bias
        conf_weights = torch.softmax(conf_logits, dim=1)
        fused_conf = (conf_weights * bev_views).sum(dim=1)

        if snr_db is None:
            gate = fused_attn.new_full((B, C, 1, 1), 0.5)
        else:
            snr_norm = (snr_db / 20.0).clamp(-1.5, 1.5)
            gate = self.snr_gate(snr_norm).view(B, C, 1, 1)

        fused = gate * fused_attn + (1.0 - gate) * fused_conf
        return fused, attn_map, conf_weights


class EnhancedMapHead(nn.Module):
    def __init__(self, in_channels):
        super(EnhancedMapHead, self).__init__()
        hidden = 256
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )

        self.branch_d1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.branch_d2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.branch_d4 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True)
        )

        self.merge = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.stem(x)
        b1 = self.branch_d1(x)
        b2 = self.branch_d2(x)
        b4 = self.branch_d4(x)
        fused = torch.cat([b1, b2, b4], dim=1)
        return self.merge(fused)





class PerspTransDetector(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.args = args
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        self.tau_2 = args.tau_2
        self.tau_1 = args.tau_1
        self.drop_prob = args.drop_prob
        self.method = getattr(args, 'method', 'baseline')
        self.disable_quantization = getattr(args, 'disable_quantization', False)
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

        # Store intrinsic and extrinsic matrices without adding errors
        self.intrinsic_matrices = dataset.base.intrinsic_matrices
        self.extrinsic_matrices = dataset.base.extrinsic_matrices
        self.worldgrid2worldcoord_mat = dataset.base.worldgrid2worldcoord_mat
        self.translation_error = 0.0
        self.rotation_error = 0.0
        self.error_camera = [ ]
        self.epoch_thres = 10
        
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat)
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        # img
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        self.img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        self.map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))

        # projection matrices: img feat -> map feat
        self.proj_mats = [torch.from_numpy(self.map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ self.img_zoom_mat)
                          for cam in range(self.num_cam)]

        base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])
        split = 7
        self.base_pt1 = base[:split].to('cuda:1')
        self.base_pt2 = base[split:].to('cuda:0')

        self.channel = 8 #self.args.compressed_channel
        self.channel_factor = 4

        # # 使用轻量级优先级网络
        # self.priority_network = LightweightPriorityNetwork(
        #     num_cam=self.num_cam,
        #     tau_1=self.tau_1,
        #     channel=self.channel
        # ).to('cuda:0')

        self.feature_extraction = nn.Conv2d(512, self.channel, 1).to("cuda:0")

        # self.temporal_entropy_model = nn.Sequential(
        #                             nn.Conv2d(self.tau_2 * self.channel, 64, kernel_size = 3, stride = 1, padding= 1),
        #                             nn.ReLU(),
        #                             nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding= 1),
        #                             nn.ReLU(),
        #                             nn.Conv2d(64, 2 * self.channel, kernel_size = 3, stride = 1, padding= 1),
        #                             ).to('cuda:0')

        # Use attention to model the temporal entropy
        self.temporal_entropy_model = TemporalEntropyModel(self.tau_2, self.channel).to('cuda:0')

        # self.temporal_fusion_module = nn.Sequential(nn.Conv2d(self.channel * self.num_cam * (self.tau_1 + 1 )+ 2, 512, 3, padding=1), nn.ReLU(),
        #                                     nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
        #                                     nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to('cuda:0')

        # self.temporal_fusion_module = TemporalFusionModule(num_channels=self.channel, num_cameras=self.num_cam, num_frames=self.tau_1 + 1).to('cuda:0')
        
        self.temporal_fusion_module = AdaptiveTemporalFusionModule(
            in_channels=self.channel * self.num_cam * (self.tau_1 + 1) + 2,num_cam=self.num_cam,tau_1=self.tau_1).to('cuda:0')


       
        self.proposed_tx_channels = (self.tau_1 + 1) * self.channel
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
            num_heads=self.cross_view_heads
        ).to('cuda:0')

        self.proposed_map_head = EnhancedMapHead(self.proposed_tx_channels + 2).to('cuda:0')

        pretrained_model_path = self.args.model_path
        if not pretrained_model_path or (not os.path.exists(pretrained_model_path)):
            raise FileNotFoundError(f"model_path not found: {pretrained_model_path}")

        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        model_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

        missing_keys, unexpected_keys = self.load_state_dict(model_dict, strict=False)
        print(f"Loaded checkpoint from {pretrained_model_path}")
        print(f"Checkpoint load summary: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")

        for param in self.base_pt1.parameters():
            param.requires_grad = False
        for param in self.base_pt2.parameters():
            param.requires_grad = False
        for param in self.feature_extraction.parameters():
            param.requires_grad = False

    def process_features_with_temporal_fusion(self, cam_feature, coord_map, is_training=True):
        B, _, H, W = cam_feature.shape  # 获取 batch size, 高度和宽度

        # # 调试：打印重要信息
        # print(f"Camera Feature Shape: {cam_feature.shape}")
        # print(f"num_cam: {self.num_cam}, tau_1: {self.tau_1}")

        # 将相机的 tau_1=0 帧扩展成与原始 world_features 形状一致
        expanded_cam_feature = cam_feature.repeat(1, self.channel * self.num_cam * (self.tau_1 + 1), 1, 1)  # [B, self.channel * self.num_cam * (self.tau_1 + 1), H, W]

        # 拼接 world_features 和 coord_map
        world_features_with_coord = torch.cat([expanded_cam_feature, coord_map.repeat([B, 1, 1, 1]).to(cam_feature.device)], dim=1)

        # 创建一个大小为 [B, self.num_cam * (self.tau_1 + 1)] 的 mask，全部值为 1
        mask = torch.ones(B, self.num_cam * (self.tau_1 + 1), dtype=torch.float32, device=cam_feature.device)

        # 传入 temporal_fusion_module 进行处理
        map_result = self.temporal_fusion_module(world_features_with_coord, mask)

        return map_result



    def save_map_result_images(self, world_features, coord_map, save_dir):
        B, C, H, W = world_features.shape  # 获取批次大小, 通道数, 高度和宽度

        # 确保保存路径存在
        os.makedirs(save_dir, exist_ok=True)

        # 遍历每个相机，处理 tau_1=0 的数据
        for cam_num in range(self.num_cam):
            # 提取当前 camera 的 tau_1=0 数据
            cam_feature = world_features[:, cam_num, :, :].unsqueeze(1)  # 提取相机的 tau_1=0 数据 [B, 1, H, W]

            # 扩展特征并处理，传入 temporal_fusion_module
            map_result = self.process_features_with_temporal_fusion(cam_feature, coord_map, is_training=self.training)

            # 保存每个相机的 map_result 为图像
            for b in range(B):
                result_image = map_result[b, 0, :, :].detach().cpu().numpy()  # 使用 detach() 使张量与计算图分离
                image_path = os.path.join(save_dir, f"batch_{b}_camera_{cam_num}_map_result.png")
                plt.imsave(image_path, result_image, cmap='gray')
                # print(f"保存相机 {cam_num} 的 map_result 到: {image_path}")




    def feature_extraction_step(self, imgs):

        B, N, C, H, W = imgs.shape

        imgs = torch.reshape(imgs,(B * N, C, H, W))

        img_feature = self.base_pt1(imgs.to('cuda:1'))
        img_feature = self.base_pt2(img_feature.to('cuda:0'))
        img_feature = self.feature_extraction(img_feature)
        if not self.disable_quantization:
            img_feature = torch.round(img_feature)


        _, C, H, W = img_feature.shape

        img_feature = torch.reshape(img_feature,(B, N, C, H, W))

        return img_feature # (B,N,channel,90,160)

    def sample_snr_db(self, batch_size, device):
        if self.training:
            return torch.empty(batch_size, 1, device=device).uniform_(self.snr_min_db, self.snr_max_db)
        return torch.full((batch_size, 1), self.test_snr_db, device=device)

    def sample_rate_scale(self, snr_db):
        snr_norm = (snr_db / 20.0).clamp(-1.5, 1.5)
        rate = 0.55 + 0.45 * torch.sigmoid(2.0 * snr_norm)
        return rate.clamp(0.25, 1.0)

    def forward_proposed(self, imgs_list):
        B, T, N, C, H, W = imgs_list.shape
        assert N == self.num_cam
        tau = max(self.tau_1, self.tau_2)

        imgs_list_feature = []
        for i in range(tau + 1):
            imgs_feature = self.feature_extraction_step(imgs_list[:, i])
            imgs_list_feature.append(imgs_feature.unsqueeze(dim=1))
        imgs_list_feature = torch.cat(imgs_list_feature, dim=1)

        to_be_transmitted_feature = imgs_list_feature[:, self.tau_2]
        conditional_features = imgs_list_feature[:, :self.tau_2]
        conditional_features = torch.swapaxes(conditional_features, 1, 2)
        conditional_features = torch.reshape(conditional_features, (B, N, self.tau_2 * self.channel, 90, 160))
        conditional_features = torch.reshape(conditional_features, (B * N, self.tau_2 * self.channel, 90, 160))

        gaussian_params = self.temporal_entropy_model(conditional_features)
        gaussian_params = torch.reshape(gaussian_params, (B, N, 2 * self.channel, 90, 160))
        scales_hat, means_hat = gaussian_params.chunk(2, dim=2)
        feature_likelihoods = GaussianLikelihoodEstimation(to_be_transmitted_feature, scales_hat, means=means_hat)
        entropy_bits_loss = (torch.log(feature_likelihoods).sum() / (-math.log(2)))

        feature4prediction = imgs_list_feature[:, -(self.tau_1 + 1):]
        feature4prediction = torch.swapaxes(feature4prediction, 1, 2)
        feature4prediction = torch.reshape(feature4prediction, (B, N, self.proposed_tx_channels, 90, 160))
        feature4prediction = torch.reshape(feature4prediction, (B * N, self.proposed_tx_channels, 90, 160))
        feature4prediction = F.interpolate(feature4prediction, size=self.upsample_shape, mode='bilinear')
        feature4prediction = torch.reshape(feature4prediction, (B, N, self.proposed_tx_channels, 270, 480))

        flat_features = feature4prediction.reshape(B, N * self.proposed_tx_channels, 270, 480)
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
            tx_mask = torch.ones(B, self.num_cam, dtype=flat_features.dtype, device=flat_features.device)
        feature4prediction = flat_features.view(B, N, self.proposed_tx_channels, 270, 480)

        snr_db = self.sample_snr_db(B, feature4prediction.device)
        if self.ablate_no_csi:
            snr_db = torch.full_like(snr_db, self.test_snr_db)

        global_rate_scale = self.sample_rate_scale(snr_db) if self.rate_aware_training else torch.ones(B, 1, device=snr_db.device)

        bev_views = []
        view_rate_scales = []
        view_snr_norm = (snr_db / 20.0).clamp(-1.5, 1.5)
        for cam in range(self.num_cam):
            cam_keep = tx_mask[:, cam:cam + 1]
            cam_feature = feature4prediction[:, cam] * cam_keep.view(B, 1, 1, 1)

            if self.ablate_no_jscc:
                tx_feat = cam_feature
                rate_used = cam_keep.view(B)
            else:
                if self.ablate_no_analog_channel:
                    tx_feat, _, rate_used = self.proposed_jscc(cam_feature, torch.full_like(snr_db, 100.0), rate_scale=global_rate_scale)
                else:
                    tx_feat, _, rate_used = self.proposed_jscc(cam_feature, snr_db, rate_scale=global_rate_scale)
                rate_used = rate_used * cam_keep.view(B)

            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to('cuda:0')
            bev_views.append(kornia.geometry.transform.warp_perspective(tx_feat.to('cuda:0'), proj_mat, self.reducedgrid_shape))
            view_rate_scales.append(rate_used)

        bev_views = torch.stack(bev_views, dim=1)
        view_rate_tensor = torch.stack(view_rate_scales, dim=1)
        view_reliability = torch.stack([
            view_snr_norm.repeat(1, self.num_cam),
            tx_mask,
            view_rate_tensor,
        ], dim=-1)

        if self.ablate_no_cross_view:
            fused_bev = bev_views.mean(dim=1)
            conf_weights = torch.full((B, self.num_cam, 1, 1, 1), 1.0 / self.num_cam, device=bev_views.device, dtype=bev_views.dtype)
        else:
            fused_bev, _, conf_weights = self.proposed_cross_view(bev_views, snr_db, view_reliability=view_reliability)

        coord = self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')
        map_result = self.proposed_map_head(torch.cat([fused_bev, coord], dim=1))

        entropy_kb = entropy_bits_loss / 8 / 1024
        keep_ratio = tx_mask.mean()
        rate_ratio = view_rate_tensor.mean()
        tx_kb_proxy = self.target_comm_kb * keep_ratio * rate_ratio

        if self.training and self.rate_aware_training and self.rate_view_dropout_prob > 0:
            random_drop = (torch.rand(B, self.num_cam, device=bev_views.device) > self.rate_view_dropout_prob).float()
            random_drop = torch.maximum(random_drop, tx_mask)
            random_drop = random_drop / (random_drop.sum(dim=1, keepdim=True) + 1e-6)
            weak_fused = (bev_views * random_drop.view(B, self.num_cam, 1, 1, 1)).sum(dim=1)
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
            map_results = torch.zeros(B, self.num_cam, map_result.shape[-2], map_result.shape[-1], device=map_result.device)

        return map_result, bits_loss, map_results

    def forward(self, imgs_list, visualize=False):
        if self.method == 'proposed_jscc':
            return self.forward_proposed(imgs_list)


        imgs_list_feature = []
        B, T ,N, C, H, W = imgs_list.shape
        assert N == self.num_cam
        world_features = []
        is_training=self.training
        # feature_bits_loss = 0
        # bits_loss = 0
        tau = max(self.tau_1, self.tau_2)
        for i in range(tau + 1):
            imgs  = imgs_list[:,i]
            imgs_feature = self.feature_extraction_step(imgs) # (B,N,channel,90,160)
            imgs_feature = imgs_feature.unsqueeze(dim=1) # (B, 1, N,channel,90,160)
            imgs_list_feature.append(imgs_feature)
        imgs_list_feature = torch.cat(imgs_list_feature, dim = 1) # (B, T, N,channel,90,160)
        #print("size", imgs_list_feature.size())
        assert T == imgs_list_feature.size()[1]

        to_be_transmitted_feature = imgs_list_feature[:,self.tau_2] # (B, N, channel, 90, 160)
        # print("to_be_transmitted_feature size", to_be_transmitted_feature.size())

        conditional_features = imgs_list_feature[:,:self.tau_2] # (B, self.tau2, N, channel 90, 160)
        conditional_features = torch.swapaxes(conditional_features, 1, 2) # (B, N, self.tau2, channel 90, 160)
        conditional_features = torch.reshape(conditional_features, (B,N, self.tau_2 * self.channel, 90, 160))
        conditional_features = torch.reshape(conditional_features, (B * N, self.tau_2 * self.channel, 90, 160))

        gaussian_params = self.temporal_entropy_model(conditional_features) # (B * N, 2 * self.channel, 90, 160)
        gaussian_params = torch.reshape(gaussian_params, (B, N, 2 * self.channel, 90, 160))
        scales_hat, means_hat = gaussian_params.chunk(2, dim = 2) # (B, N, self.channel, 90, 160)
        feature_likelihoods = GaussianLikelihoodEstimation(to_be_transmitted_feature, scales_hat, means=means_hat)
        # print("feature_likelihoods size", feature_likelihoods.size())
        # print("feature_likelihoods", feature_likelihoods)
        bits_loss = (torch.log(feature_likelihoods).sum() / (-math.log(2)))

        feature4prediction = imgs_list_feature[:,-(self.tau_1+1):] # (B, (self.tau_1 +1), N, channel, 90, 160)
        feature4prediction = torch.swapaxes(feature4prediction, 1, 2) # (B, N, (self.tau_1 +1), channel, 90, 160)
        feature4prediction = torch.reshape(feature4prediction, (B,N, (self.tau_1 +1) * self.channel, 90, 160)) # (B, N, (self.tau_1 +1) * channel, 90, 160)
        feature4prediction = torch.reshape(feature4prediction, (B * N, (self.tau_1 +1) * self.channel, 90, 160)) # (B, N, (self.tau_1 +1) * channel, 90, 160)
        feature4prediction = F.interpolate(feature4prediction, size = self.upsample_shape, mode='bilinear') # (B, N, (self.tau_1 +1) * channel, H, W)
        feature4prediction = torch.reshape(feature4prediction, (B, N, (self.tau_1 +1) * self.channel, 270, 480)) # (B, N, (self.tau_1 +1) * channel, 270, 480)

        # Read the epoch from the local log file (fallback to 0 if unavailable)
        epoch = 0
        if os.path.exists('epoch.log'):
            try:
                with open('epoch.log', 'r') as f:
                    epoch = int(f.read().strip())
            except Exception:
                epoch = 0


        # Only update error parameters for epoch > 10 and during testing phase
        if epoch > self.epoch_thres and not self.training:
            print(f"Epoch: {epoch}, Reading CSV for test parameters...")

            csv_path = os.path.join('temp', 'Calibration', 'calibration_test_rotation_error.csv')
            if not os.path.exists(csv_path):
                print(f"Calibration CSV not found: {csv_path}, skip error injection.")
                self.error_camera = []
                self.translation_error = 0.0
                self.rotation_error = 0.0
            else:
                csv_data = pd.read_csv(csv_path)
                if "Epoch" not in csv_data.columns or csv_data[csv_data["Epoch"] == epoch].empty:
                    print(f"No calibration row for epoch {epoch}, skip error injection.")
                    self.error_camera = []
                    self.translation_error = 0.0
                    self.rotation_error = 0.0
                else:
                    test_params = csv_data[csv_data["Epoch"] == epoch].iloc[0]
                    translation_error = test_params['Translation Error']
                    rotation_error = test_params['Rotation Error']
                    error_camera = test_params['error_camera']
                    print(f"Loaded params - Translation Error: {translation_error}, Rotation Error: {rotation_error}, Error Camera: {error_camera}")

                    # Store the parameters in the model instance
                    self.translation_error = translation_error
                    self.rotation_error = rotation_error
                    self.error_camera = error_camera

                    # Convert error_camera string to list of integers
                    if isinstance(self.error_camera, str):
                        self.error_camera = [int(cam.strip()) for cam in self.error_camera.split(',') if cam.strip().isdigit()]
                    elif isinstance(self.error_camera, (int, float)):
                        self.error_camera = [int(self.error_camera)]
                    elif not isinstance(self.error_camera, list):
                        self.error_camera = []
                    print(f"Error Camera List: {self.error_camera}")

                    # Apply errors only during the test phase
                    extrinsic_matrices = np.array(self.extrinsic_matrices, dtype=np.float64)

                    for cam in self.error_camera:
                        cam = int(cam)
                        if cam >= len(extrinsic_matrices):
                            raise IndexError(f"Camera index {cam} is out of bounds for extrinsic_matrices.")

                        R = extrinsic_matrices[cam][:, :3]
                        T = extrinsic_matrices[cam][:, 3]

                        # Apply perturbations
                        rotation_perturbation = np.random.randn(3, 3) * self.rotation_error
                        translation_perturbation = np.random.randn(3) * self.translation_error * 100
                        perturbed_R = R * (1 + rotation_perturbation)
                        perturbed_T = T * (1 + translation_perturbation)

                        extrinsic_matrices[cam][:, :3] = perturbed_R
                        extrinsic_matrices[cam][:, 3] = perturbed_T

                    self.extrinsic_matrices = extrinsic_matrices
                    print(f"Applied translation error {self.translation_error} and rotation error {self.rotation_error} to cameras {self.error_camera}")


        # Only update matrices in test phase and after epoch 10
        if not self.training and epoch > self.epoch_thres:
            imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(
                self.intrinsic_matrices,
                self.extrinsic_matrices,
                self.worldgrid2worldcoord_mat
            )

            # Update projection matrices with errors
            self.proj_mats = [torch.from_numpy(self.map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ self.img_zoom_mat)
                            for cam in range(self.num_cam)]
            print("Updated projection matrices for testing phase.")


        for cam in range(self.num_cam):
            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to('cuda:0')
            if not self.training and epoch > self.epoch_thres:
                # Update projection matrices with errors
                proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to('cuda:0')
                print(f"Updated projection matrix for camera {cam}.")
                print(f"Projection Matrix: {proj_mat}")
            world_feature = kornia.geometry.transform.warp_perspective(feature4prediction[:,cam].to('cuda:0'), proj_mat, self.reducedgrid_shape)
            #print(world_feature.size()) # （B, self.tau_1 * channel, H, W）
            world_features.append(world_feature.to('cuda:0'))
        
        # 确保 world_features 是一个列表，然后将其拼接为张量
        world_features = torch.cat(world_features, dim=1)  # 拼接后变成张量

        # 假设已经在模型中定义了该函数，调用时可以这样做
        save_dir = os.path.join('temp', 'map_res')
        self.save_map_result_images(world_features, self.coord_map, save_dir)


        # 使用优先级向量进行随机丢帧
        world_features, mask = random_drop_frame_with_priority(world_features, self.num_cam, self.tau_1, self.channel, self.drop_prob,is_training=self.training)
        # print("world_features size", world_features.size()) # torch.Size([B, self.channel * self.num_cam * (self.tau_1 + 1), 120, 360])
        # print("mask size", mask.size()) # torch.Size([B, self.num_cam * (self.tau_1 + 1)])
        
        # world_features, mask = random_drop_frame(world_features, self.num_cam, self.tau_1, self.channel,self.drop_prob)

        # print("world_features size", world_features.size()) # torch.Size([B, self.channel * self.num_cam * (self.tau_1 + 1), 120, 360])
        # print("mask size", mask.size()) # torch.Size([B, self.num_cam * (self.tau_1 + 1)])

        # 使用 torch.cat 来拼接 world_features 和 coord_map

        world_features = torch.cat([world_features, self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        map_result = self.temporal_fusion_module(world_features, mask)
        # print("map_result size", map_result.size())

        # print("world_features size", world_features[0].size()) # torch.Size([B, self.channel *  (self.tau_1 +1), 120, 360])
        # world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        # # print("self.coord_map.repeat([B, 1, 1, 1])", self.coord_map.repeat([B, 1, 1, 1]).size())
        # world_features = world_features.to('cuda:0') # （B, self.channel * self.num_cam * (self.tau_1 + 1 )+ 2, H, W）,H=120,W=360

        # map_result = self.temporal_fusion_module(world_features)
        # print("map_result size", map_result.size())

        map_results = []  # 用于保存每个相机的 map_result

        bits_loss = bits_loss / 8 / 1024

        # 如果是测试阶段，计算每个相机的 map_res
        if not self.training:
            map_results = []  # 用于保存每个相机的 map_result
            for cam_num in range(self.num_cam):
                # 提取相机的特征
                cam_feature = world_features[:, cam_num, :, :].unsqueeze(1)  # 提取相机的 tau_1=0 数据 [B, 1, H, W]

                # 扩展特征并处理，传入 temporal_fusion_module
                map_result_single = self.process_features_with_temporal_fusion(cam_feature, self.coord_map, is_training=self.training)
                map_results.append(map_result_single)

            # 将所有相机的 map_res 拼接
            map_results = torch.cat(map_results, dim=1)  # [B, num_cam, H, W]
        else:
            # 训练阶段，将 map_results 设置为全 0
            map_results = torch.zeros(B, self.num_cam, H, W, device=world_features.device)


        return map_result, bits_loss, map_results #0#, imgs_result

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
            pass
        return projection_matrices


    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret
