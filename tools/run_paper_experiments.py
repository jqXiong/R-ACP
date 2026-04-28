import argparse
import csv
import datetime
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torchvision.transforms as T

from multiview_detector.datasets import Wildtrack, sequenceDataset4phase2
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.utils.codec_baselines import (
    TensorImageCodec,
    apply_codec_packet_loss_to_batch,
)
from multiview_detector.utils.nms import nms


DEFAULT_MAIN_ARGS = {
    'cls_thres': 0.4,
    'num_workers': 8,
    'batch_size': 1,
    'epochs': 10,
    'train_epochs': -1,
    'lr': 0.1,
    'weight_decay': 5e-4,
    'momentum': 0.5,
    'log_interval': 20,
    'seed': 1,
    'tau_1': 0,
    'tau_2': 1,
    'dataset_path': './Data/Wildtrack',
    'model_path': '',
    'drop_prob': 0.0,
    'method': 'baseline',
    'refine_keep_cameras': -1,
    'refine_keep_ratio': 1.0,
    'refine_min_keep_cameras': 1,
    'refine_score_mode': 'current',
    'refine_score_noise_std': 0.0,
    'refine_weighted_entropy': False,
    'refine_enable_token_drop': False,
    'refine_soft_weighting': False,
    'refine_adaptive_keep_margin': 0.0,
    'refine_scale_min': 0.6,
    'refine_scale_max': 1.0,
    'refine_apply_entropy_scale': False,
    'refine_channel_keep_ratio': 1.0,
    'refine_channel_min_keep': 1,
    'refine_channel_drop_floor': 0.0,
    'refine_snr_aware': False,
    'refine_snr_reference_db': 10.0,
    'refine_low_snr_camera_bonus': 0,
    'refine_low_snr_channel_bonus': 0.0,
    'refine_low_snr_drop_floor': 0.0,
    'refine_low_snr_scale_boost': 0.0,
    'refine_use_cross_view_fusion': False,
    'refine_use_bev_attention': False,
    'refine_strong_head_weight': 0.0,
    'disable_quantization': False,
    'jscc_channel_type': 'rayleigh',
    'jscc_latent_channels': -1,
    'snr_min_db': 0.0,
    'snr_max_db': 20.0,
    'test_snr_db': 20.0,
    'cross_view_heads': 4,
    'jscc_csi_gain_scale': 0.6,
    'jscc_importance_gain_scale': 1.0,
    'jscc_low_snr_disable_csi_threshold': 0.0,
    'ablate_no_jscc': False,
    'ablate_no_csi': False,
    'ablate_no_analog_channel': False,
    'ablate_no_cross_view': False,
    'rate_aware_training': False,
    'target_comm_kb': 28.5,
    'min_keep_per_camera': 1,
    'keep_latest_token': True,
    'frame_dropout_noise_std': 0.05,
    'lambda_consistency': 0.03,
    'rate_view_dropout_prob': 0.2,
    'snr_sweep': '',
    'snr_sweep_resume': False,
    'test_only': True,
    'save_prefix': '',
    'early_stop_patience': 6,
    'early_stop_min_delta': 0.05,
    'early_stop_min_epochs': 12,
    'exp_name': '',
}


METHOD_SPECS = {
    'baseline': {'model_key': 'baseline', 'codec': None, 'internal_packet_loss': True},
    'jpeg': {'model_key': 'baseline', 'codec': 'jpeg', 'internal_packet_loss': False},
    'h264': {'model_key': 'baseline', 'codec': 'h264', 'internal_packet_loss': False},
    'h265': {'model_key': 'baseline', 'codec': 'h265', 'internal_packet_loss': False},
    'av1': {'model_key': 'baseline', 'codec': 'av1', 'internal_packet_loss': False},
    'full': {'model_key': 'full', 'codec': None, 'internal_packet_loss': True},
}


def load_checkpoint_args(checkpoint_path: str) -> Dict:
    payload = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(payload, dict):
        return payload.get('args', {}) or {}
    return {}


def build_runtime_args(checkpoint_path: str, overrides: Dict) -> SimpleNamespace:
    merged = dict(DEFAULT_MAIN_ARGS)
    merged.update(load_checkpoint_args(checkpoint_path))
    merged.update(overrides)
    return SimpleNamespace(**merged)


def build_dataloaders(args, tau: int):
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize])
    base = Wildtrack(os.path.expanduser(args.dataset_path))
    test_set = sequenceDataset4phase2(base, tau=tau, train=False, transform=transform, grid_reduce=4)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return test_set, test_loader


def build_model_for_method(runtime_args, dataset):
    model = PerspTransDetector(dataset, runtime_args)
    model.enable_calibration_error = False
    model.enable_map_visual_dump = False
    model.test_snr_db = runtime_args.test_snr_db
    return model


def prepare_models(args, dataset):
    baseline_args = build_runtime_args(
        args.baseline_ckpt,
        {
            'model_path': args.baseline_ckpt,
            'method': 'baseline',
            'drop_prob': 0.0,
            'dataset_path': args.dataset_path,
            'test_snr_db': args.test_snr_db,
        },
    )
    full_args = build_runtime_args(
        args.full_ckpt,
        {
            'model_path': args.full_ckpt,
            'method': 'baseline_refined',
            'drop_prob': 0.0,
            'dataset_path': args.dataset_path,
            'test_snr_db': args.test_snr_db,
        },
    )
    models = {
        'baseline': build_model_for_method(baseline_args, dataset),
        'full': build_model_for_method(full_args, dataset),
    }
    return models


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def extract_frame_value(frame_item):
    if isinstance(frame_item, torch.Tensor):
        return int(frame_item.item())
    return int(frame_item)


def compute_aopt(frame_stats: List[Dict], capacity_kbps: float, lambda_camera: float, min_targets: int) -> float:
    delta_t = 1.0 / max(lambda_camera, 1e-6)
    total = 0.0
    for row in frame_stats:
        perceived_targets = row['num_targets']
        if perceived_targets < min_targets:
            continue
        d_total = row['comm_kb'] / max(capacity_kbps, 1e-6)
        total += perceived_targets * (delta_t ** 2 + d_total)
    return total / max(len(frame_stats), 1)


def run_single_evaluation(
    method_name: str,
    model,
    data_loader,
    criterion,
    cls_thres: float,
    output_dir: str,
    packet_loss_rate: float,
    codec_runner: Optional[TensorImageCodec],
    collect_frame_stats: bool,
    seed: int,
    max_eval_batches: int,
):
    spec = METHOD_SPECS[method_name]
    ensure_dir(output_dir)
    res_fpath = os.path.join(output_dir, 'test.txt')
    all_res_fpath = os.path.join(output_dir, 'all_res.txt')
    gt_fpath = data_loader.dataset.gt_fpath

    model.eval()
    if spec['internal_packet_loss']:
        model.drop_prob = packet_loss_rate
    else:
        model.drop_prob = 0.0

    losses = 0.0
    communication_costs = []
    all_res_list = []
    frame_stats = []
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for batch_idx, (data, map_gt, _, frame) in enumerate(data_loader):
            if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                break

            if codec_runner is not None:
                eval_data, comm_kb = apply_codec_packet_loss_to_batch(
                    data,
                    frame,
                    codec_runner=codec_runner,
                    packet_loss_rate=packet_loss_rate,
                    concealment='previous',
                    transmitted_index=-1,
                    rng=rng,
                )
            else:
                eval_data = data
                comm_kb = None

            map_res, bits_loss, _ = model(eval_data)
            if comm_kb is None:
                comm_kb = float(bits_loss.item())
            communication_costs.append(comm_kb)

            loss = criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel)
            losses += float(loss.item())

            map_grid_res = map_res.detach().cpu()
            batch_size = map_grid_res.shape[0]
            per_sample_comm = comm_kb / max(batch_size, 1)
            for b in range(batch_size):
                frame_value = extract_frame_value(frame[b] if isinstance(frame, torch.Tensor) and frame.ndim > 0 else frame)
                single_map = map_grid_res[b].squeeze()
                score_values = single_map[single_map > cls_thres].unsqueeze(1)
                grid_ij = (single_map > cls_thres).nonzero()
                if data_loader.dataset.base.indexing == 'xy' and grid_ij.numel() > 0:
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij

                if score_values.numel() > 0:
                    frame_res = torch.cat(
                        [
                            torch.ones_like(score_values) * frame_value,
                            grid_xy.float() * data_loader.dataset.grid_reduce,
                            score_values,
                        ],
                        dim=1,
                    )
                    all_res_list.append(frame_res)

                    positions, scores = frame_res[:, 1:3], frame_res[:, 3]
                    ids, count = nms(positions, scores, 20, np.inf)
                    target_count = int(count)
                else:
                    target_count = 0

                if collect_frame_stats:
                    frame_stats.append(
                        {
                            'frame_id': frame_value,
                            'num_targets': target_count,
                            'comm_kb': per_sample_comm,
                        }
                    )

    if all_res_list:
        all_res_tensor = torch.cat(all_res_list, dim=0)
        np.savetxt(all_res_fpath, all_res_tensor.numpy(), '%.8f')
        reduced_res = []
        for frame_num in np.unique(all_res_tensor[:, 0]):
            res = all_res_tensor[all_res_tensor[:, 0] == frame_num, :]
            positions, scores = res[:, 1:3], res[:, 3]
            ids, count = nms(positions, scores, 20, np.inf)
            reduced_res.append(torch.cat([torch.ones([count, 1]) * frame_num, positions[ids[:count], :]], dim=1))
        res_array = torch.cat(reduced_res, dim=0).numpy() if reduced_res else np.empty([0, 3], dtype=np.float32)
    else:
        np.savetxt(all_res_fpath, np.empty([0, 4], dtype=np.float32), '%.8f')
        res_array = np.empty([0, 3], dtype=np.float32)
    np.savetxt(res_fpath, res_array, '%d')

    eval_recall, eval_precision, moda, modp = evaluate(
        os.path.abspath(res_fpath),
        os.path.abspath(gt_fpath),
        data_loader.dataset.base.__name__,
    )

    frame_stats_path = None
    if collect_frame_stats:
        frame_stats_path = os.path.join(output_dir, 'frame_stats.csv')
        with open(frame_stats_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame_id', 'num_targets', 'comm_kb'])
            writer.writeheader()
            writer.writerows(frame_stats)

    return {
        'method': method_name,
        'packet_loss_rate': packet_loss_rate,
        'loss': losses / max(len(communication_costs), 1),
        'moda': moda,
        'modp': modp,
        'eval_precision': eval_precision,
        'eval_recall': eval_recall,
        'comm_kb': float(np.mean(communication_costs)) if communication_costs else 0.0,
        'frame_stats': frame_stats,
        'frame_stats_path': frame_stats_path,
    }


def run_packet_loss_experiment(args, models, data_loader, criterion):
    output_csv = os.path.join(args.output_dir, 'packet_loss_summary.csv')
    rows = []
    codec_runners = {}

    for method_name in args.methods:
        spec = METHOD_SPECS[method_name]
        model = models[spec['model_key']]
        codec_runner = None
        if spec['codec'] is not None:
            codec_runner = codec_runners.get(spec['codec'])
            if codec_runner is None:
                codec_runner = TensorImageCodec(
                    spec['codec'],
                    ffmpeg_bin=args.ffmpeg_bin,
                    jpeg_quality=args.jpeg_quality,
                    h264_crf=args.h264_crf,
                    h265_crf=args.h265_crf,
                    av1_crf=args.av1_crf,
                )
                codec_runners[spec['codec']] = codec_runner

        for loss_rate in args.packet_loss_rates:
            experiment_dir = os.path.join(args.output_dir, method_name, f'packet_loss_{str(loss_rate).replace(".", "p")}')
            result = run_single_evaluation(
                method_name=method_name,
                model=model,
                data_loader=data_loader,
                criterion=criterion,
                cls_thres=args.cls_thres,
                output_dir=experiment_dir,
                packet_loss_rate=loss_rate,
                codec_runner=codec_runner,
                collect_frame_stats=False,
                seed=args.seed + int(loss_rate * 1000),
                max_eval_batches=args.max_eval_batches,
            )
            rows.append(
                {
                    'method': method_name,
                    'packet_loss_rate': loss_rate,
                    'loss': result['loss'],
                    'moda': result['moda'],
                    'modp': result['modp'],
                    'eval_precision': result['eval_precision'],
                    'eval_recall': result['eval_recall'],
                    'comm_kb': result['comm_kb'],
                    'result_dir': experiment_dir,
                }
            )

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'method',
                'packet_loss_rate',
                'loss',
                'moda',
                'modp',
                'eval_precision',
                'eval_recall',
                'comm_kb',
                'result_dir',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote packet-loss comparison to {output_csv}')


def run_aopt_experiment(args, models, data_loader, criterion):
    detail_csv = os.path.join(args.output_dir, 'aopt_summary.csv')
    rows = []
    codec_runners = {}

    for method_name in args.methods:
        spec = METHOD_SPECS[method_name]
        model = models[spec['model_key']]
        codec_runner = None
        if spec['codec'] is not None:
            codec_runner = codec_runners.get(spec['codec'])
            if codec_runner is None:
                codec_runner = TensorImageCodec(
                    spec['codec'],
                    ffmpeg_bin=args.ffmpeg_bin,
                    jpeg_quality=args.jpeg_quality,
                    h264_crf=args.h264_crf,
                    h265_crf=args.h265_crf,
                    av1_crf=args.av1_crf,
                )
                codec_runners[spec['codec']] = codec_runner

        experiment_dir = os.path.join(args.output_dir, method_name, 'capacity_sweep')
        result = run_single_evaluation(
            method_name=method_name,
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            cls_thres=args.cls_thres,
            output_dir=experiment_dir,
            packet_loss_rate=args.aopt_packet_loss_rate,
            codec_runner=codec_runner,
            collect_frame_stats=True,
            seed=args.seed,
            max_eval_batches=args.max_eval_batches,
        )

        for capacity in args.aopt_capacities:
            rows.append(
                {
                    'method': method_name,
                    'capacity_kbps': capacity,
                    'aopt': compute_aopt(
                        result['frame_stats'],
                        capacity_kbps=capacity,
                        lambda_camera=args.lambda_camera,
                        min_targets=args.aopt_min_targets,
                    ),
                    'moda': result['moda'],
                    'modp': result['modp'],
                    'eval_precision': result['eval_precision'],
                    'eval_recall': result['eval_recall'],
                    'comm_kb': result['comm_kb'],
                    'frame_stats_path': result['frame_stats_path'],
                    'result_dir': experiment_dir,
                }
            )

    with open(detail_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'method',
                'capacity_kbps',
                'aopt',
                'moda',
                'modp',
                'eval_precision',
                'eval_recall',
                'comm_kb',
                'frame_stats_path',
                'result_dir',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote AoPT comparison to {detail_csv}')


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(',') if x.strip()]


def parse_method_list(value: str) -> List[str]:
    methods = [x.strip().lower() for x in value.split(',') if x.strip()]
    invalid = [m for m in methods if m not in METHOD_SPECS]
    if invalid:
        raise ValueError(f'Unsupported methods: {invalid}. Valid methods: {sorted(METHOD_SPECS.keys())}')
    return methods


def main():
    parser = argparse.ArgumentParser(description='Run packet-loss and AoPT comparison experiments.')
    parser.add_argument('--experiment', type=str, required=True, choices=['packet_loss', 'aopt'])
    parser.add_argument('--dataset_path', type=str, default='./Data/Wildtrack')
    parser.add_argument('--baseline_ckpt', type=str, required=True)
    parser.add_argument('--full_ckpt', type=str, required=True)
    parser.add_argument('--methods', type=str, default='baseline,jpeg,h264,h265,av1,full')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test_snr_db', type=float, default=20.0)
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg')
    parser.add_argument('--jpeg_quality', type=int, default=75)
    parser.add_argument('--h264_crf', type=int, default=28)
    parser.add_argument('--h265_crf', type=int, default=30)
    parser.add_argument('--av1_crf', type=int, default=35)
    parser.add_argument('--packet_loss_rates', type=str, default='0,0.1,0.2,0.3,0.4')
    parser.add_argument('--aopt_capacities', type=str, default='20,40,60,80,100,120')
    parser.add_argument('--aopt_packet_loss_rate', type=float, default=0.0)
    parser.add_argument('--lambda_camera', type=float, default=0.5)
    parser.add_argument('--aopt_min_targets', type=int, default=1)
    parser.add_argument('--max_eval_batches', type=int, default=-1)
    args = parser.parse_args()

    args.methods = parse_method_list(args.methods)
    args.packet_loss_rates = parse_float_list(args.packet_loss_rates)
    args.aopt_capacities = parse_float_list(args.aopt_capacities)
    if args.batch_size != 1:
        raise ValueError('These comparison scripts currently require --batch_size 1 for per-frame cost/AoPT accounting.')

    if not args.output_dir:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.output_dir = os.path.join('logs', f'{timestamp}_{args.experiment}_comparison')
    ensure_dir(args.output_dir)

    baseline_runtime_args = build_runtime_args(
        args.baseline_ckpt,
        {
            'dataset_path': args.dataset_path,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'test_snr_db': args.test_snr_db,
        },
    )
    full_runtime_args = build_runtime_args(
        args.full_ckpt,
        {
            'dataset_path': args.dataset_path,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'test_snr_db': args.test_snr_db,
        },
    )
    baseline_tau = max(baseline_runtime_args.tau_1, baseline_runtime_args.tau_2)
    full_tau = max(full_runtime_args.tau_1, full_runtime_args.tau_2)
    if baseline_tau != full_tau:
        raise ValueError(
            f'Baseline/full checkpoints use different temporal windows: baseline tau={baseline_tau}, full tau={full_tau}. '
            'Please regenerate a compatible pair before running comparison experiments.'
        )

    test_set, test_loader = build_dataloaders(baseline_runtime_args, tau=baseline_tau)
    models = prepare_models(args, test_set)
    criterion = GaussianMSE().cuda()

    print(f'Running {args.experiment} experiment')
    print(f'Methods: {args.methods}')
    print(f'Output dir: {args.output_dir}')

    if args.experiment == 'packet_loss':
        run_packet_loss_experiment(args, models, test_loader, criterion)
    else:
        run_aopt_experiment(args, models, test_loader, criterion)


if __name__ == '__main__':
    main()
