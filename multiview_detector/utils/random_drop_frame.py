import random
import torch
import os
import pandas as pd


def _reshape_tokens(features, num_cam, tau_1, channel):
    b, c, h, w = features.shape
    total_frames = num_cam * (tau_1 + 1)
    expected_c = total_frames * channel
    if c != expected_c:
        raise ValueError(f"Channel mismatch: got {c}, expected {expected_c} (=num_cam*(tau_1+1)*channel)")
    tokens = features.view(b, total_frames, channel, h, w)
    return tokens, total_frames, h, w


def random_drop_frame(features, num_cam, tau_1, channel, drop_prob):
    b, c, h, w = features.shape
    total_frames = num_cam * (tau_1 + 1)
    mask = torch.ones(b, total_frames, dtype=torch.float32, device=features.device)

    for bi in range(b):
        for cam in range(num_cam):
            if random.random() < drop_prob:
                s = cam * (tau_1 + 1)
                e = (cam + 1) * (tau_1 + 1)
                mask[bi, s:e] = 0.0

    expanded_mask = mask.unsqueeze(2).repeat(1, 1, channel).view(b, c, 1, 1).expand(-1, -1, h, w)
    masked_features = features * expanded_mask
    return masked_features, mask


def random_drop_frame_with_priority(
    features,
    num_cam,
    tau_1,
    channel,
    target_dropout_rate,
    is_training=True,
    min_keep_per_camera=1,
    keep_latest_token=True,
    score_noise_std=0.05,
):
    """
    Token-wise priority drop for (cam, tau) frames.

    Args:
        features: [B, num_cam*(tau_1+1)*channel, H, W]
    Returns:
        masked_features: same shape as features
        mask: [B, num_cam*(tau_1+1)] (1 keep / 0 drop)
    """
    tokens, total_frames, h, w = _reshape_tokens(features, num_cam, tau_1, channel)
    b = features.shape[0]

    # Priority score per (cam, tau): combine activation magnitude and variability
    mean_score = tokens.abs().mean(dim=(2, 3, 4))
    std_score = tokens.std(dim=(2, 3, 4))
    priority = mean_score + 0.5 * std_score

    if is_training and score_noise_std > 0:
        priority = priority + torch.randn_like(priority) * score_noise_std

    keep_count = max(1, int(round(total_frames * (1.0 - float(target_dropout_rate)))))
    keep_count = min(keep_count, total_frames)

    mask = torch.zeros(b, total_frames, dtype=torch.float32, device=features.device)
    frame_per_cam = tau_1 + 1

    for bi in range(b):
        forced_keep = set()

        if keep_latest_token:
            for cam in range(num_cam):
                forced_keep.add(cam * frame_per_cam + tau_1)

        if min_keep_per_camera > 0:
            for cam in range(num_cam):
                s = cam * frame_per_cam
                e = s + frame_per_cam
                cam_scores = priority[bi, s:e]
                top_local = torch.topk(cam_scores, k=min(min_keep_per_camera, frame_per_cam), largest=True).indices
                for idx in top_local.tolist():
                    forced_keep.add(s + idx)

        for idx in forced_keep:
            mask[bi, idx] = 1.0

        already_keep = int(mask[bi].sum().item())
        remain = max(0, keep_count - already_keep)

        if remain > 0:
            candidates = torch.nonzero(mask[bi] < 0.5, as_tuple=False).squeeze(1)
            if candidates.numel() > 0:
                cand_scores = priority[bi, candidates]
                top_idx = torch.topk(cand_scores, k=min(remain, candidates.numel()), largest=True).indices
                chosen = candidates[top_idx]
                mask[bi, chosen] = 1.0

    expanded_mask = mask.unsqueeze(2).repeat(1, 1, channel).view(b, total_frames * channel, 1, 1).expand(-1, -1, h, w)
    masked_features = features * expanded_mask
    return masked_features, mask


def random_drop_frame_with_priority_legacy(features, num_cam, tau_1, channel, target_dropout_rate, is_training=True):
    b, c, h, w = features.shape
    mask = torch.ones(b, num_cam * (tau_1 + 1), dtype=torch.float32, device=features.device)

    dropout_data = {
        'Batch': [],
        'Camera': [],
        'Tau': [],
        'Dropout_Probability': []
    }

    save_dir = os.path.join('temp', 'feature_temp')
    os.makedirs(save_dir, exist_ok=True)

    for bi in range(b):
        total_frames = num_cam * (tau_1 + 1)
        current_dropout_count = 0
        target_dropout_count = int(total_frames * target_dropout_rate)

        if features[bi].dim() != 3:
            raise ValueError(f"Unexpected feature shape: {features[bi].shape}")

        avg_feature_values = features[bi].mean(dim=[1, 2])
        camera_tau_list = [(cam_idx, tau) for cam_idx in range(num_cam) for tau in range(tau_1 + 1)]
        priority_values_with_indices = [(avg_feature_values[i], camera_tau_list[i]) for i in range(total_frames)]
        priority_sorted = sorted(priority_values_with_indices, key=lambda x: x[0])

        min_priority = min([x[0] for x in priority_sorted]).item()
        max_priority = max([x[0] for x in priority_sorted]).item()

        for priority_value, (cam_idx, tau) in priority_sorted:
            if current_dropout_count >= target_dropout_count:
                break

            idx = cam_idx * (tau_1 + 1) + tau
            drop_prob = (priority_value.item() - min_priority) / (max_priority - min_priority + 1e-6)

            if random.random() < drop_prob:
                mask[bi, idx] = 0
                current_dropout_count += 1

                if not is_training:
                    dropout_data['Batch'].append(bi)
                    dropout_data['Camera'].append(cam_idx)
                    dropout_data['Tau'].append(tau)
                    dropout_data['Dropout_Probability'].append(drop_prob)

    expanded_mask = mask.unsqueeze(2).repeat(1, 1, channel).view(b, c, 1, 1).expand(-1, -1, h, w)
    masked_features = features * expanded_mask

    if not is_training and dropout_data['Batch']:
        df = pd.DataFrame(dropout_data)
        df.to_excel(os.path.join(save_dir, 'dropout_data.xlsx'), index=False)

    return masked_features, mask
