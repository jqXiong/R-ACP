import random
import torch


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
