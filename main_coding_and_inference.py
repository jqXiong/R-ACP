import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import csv
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from multiview_detector.datasets import *
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.trainer import PerspectiveTrainer
if not hasattr(torch, '_six'):
    torch._six = type('six', (), {'string_classes': (str, bytes)})    

def _load_existing_snr_rows(csv_path):
    existing = {}
    if not os.path.exists(csv_path):
        return existing
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = float(row['snr_db'])
            except Exception:
                continue
            existing[s] = {
                'test_loss': float(row.get('test_loss', 'nan')),
                'test_prec': float(row.get('test_prec', 'nan')),
                'moda': float(row.get('moda', 'nan')),
                'modp': float(row.get('modp', 'nan')),
                'eval_precision': float(row.get('eval_precision', 'nan')),
                'eval_recall': float(row.get('eval_recall', 'nan')),
                'comm_kb': float(row.get('comm_kb', row.get('communication_cost', 'nan'))),
            }
    return existing


def _write_snr_rows(csv_path, rows_by_snr):
    headers = ['snr_db', 'test_loss', 'test_prec', 'moda', 'modp', 'eval_precision', 'eval_recall', 'comm_kb']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for s in sorted(rows_by_snr.keys()):
            row = rows_by_snr[s]
            writer.writerow({
                'snr_db': s,
                'test_loss': row.get('test_loss', float('nan')),
                'test_prec': row.get('test_prec', float('nan')),
                'moda': row.get('moda', float('nan')),
                'modp': row.get('modp', float('nan')),
                'eval_precision': row.get('eval_precision', float('nan')),
                'eval_recall': row.get('eval_recall', float('nan')),
                'comm_kb': row.get('comm_kb', float('nan')),
            })


def _save_checkpoint(path, model, optimizer, epoch, args, metrics=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'epoch': epoch,
        'args': vars(args),
        'metrics': metrics or {},
    }
    torch.save(payload, path)


def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    data_path = os.path.expanduser(args.dataset_path)
    base = Wildtrack(data_path)

    tau = max(args.tau_1, args.tau_2)

    train_set = sequenceDataset4phase2(base, tau = tau, train=True, transform=train_trans, grid_reduce=4)
    test_set = sequenceDataset4phase2(base, tau = tau , train=False, transform=train_trans, grid_reduce=4)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    # model
    model = PerspTransDetector(train_set, args)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)


    criterion = GaussianMSE().cuda()

    logdir = f'logs/' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    if args.exp_name:
        logdir = f"{logdir}_{args.exp_name}"

    os.makedirs(logdir, exist_ok=True)
    copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
    for script in os.listdir('.'):
        if script.split('.')[-1] == 'py':
            dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )

    print('Settings: \n', vars(args))


    trainer = PerspectiveTrainer(model, criterion, logdir, denormalize, args.cls_thres)

    max_MODA = 0
    minimum_bits_loss = 2e6
    best_moda = -1e9
    save_prefix = args.save_prefix if args.save_prefix else (args.exp_name if args.exp_name else args.method)

    total_epochs = 1 if args.test_only else args.epochs
    for epoch in tqdm.tqdm(range(1, total_epochs + 1)):
        # Log the current epoch to a file
        with open('epoch.log', 'w') as f:
            f.write(f'{epoch}\n')

        if (not args.test_only) and epoch <= 10:
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, args.log_interval, scheduler)

        print('Testing...')
        test_loss, test_prec, moda, modp, eval_precision, eval_recall, avg_bit_loss = trainer.test(
            test_loader,
            os.path.join(logdir, 'test.txt'),
            train_set.gt_fpath,
            True,
        )

        if args.snr_sweep:
            snr_values = [float(x.strip()) for x in args.snr_sweep.split(',') if x.strip()]
            sweep_csv_path = os.path.join(logdir, f'snr_sweep_epoch_{epoch}.csv')
            rows_by_snr = _load_existing_snr_rows(sweep_csv_path) if args.snr_sweep_resume else {}

            original_test_snr_db = model.test_snr_db
            try:
                for s in snr_values:
                    if args.snr_sweep_resume and s in rows_by_snr:
                        print(f'SNR sweep skip {s:.1f} dB (already exists).')
                        continue

                    model.test_snr_db = s
                    print(f'SNR sweep testing at {s:.1f} dB...')
                    test_loss_s, test_prec_s, moda_s, modp_s, eval_precision_s, eval_recall_s, bit_s = trainer.test(
                        test_loader,
                        os.path.join(logdir, f'test_snr_{str(s).replace(".", "p")}.txt'),
                        train_set.gt_fpath,
                        False,
                    )
                    rows_by_snr[s] = {
                        'test_loss': test_loss_s,
                        'test_prec': test_prec_s,
                        'moda': moda_s,
                        'modp': modp_s,
                        'eval_precision': eval_precision_s,
                        'eval_recall': eval_recall_s,
                        'comm_kb': bit_s,
                    }
                    _write_snr_rows(sweep_csv_path, rows_by_snr)
            finally:
                model.test_snr_db = original_test_snr_db

        if minimum_bits_loss > avg_bit_loss:
            minimum_bits_loss = avg_bit_loss

        if moda > max_MODA:
            max_MODA = moda

        if not args.test_only:
            ckpt_metrics = {
                'test_loss': test_loss,
                'test_prec': test_prec,
                'moda': moda,
                'modp': modp,
                'eval_precision': eval_precision,
                'eval_recall': eval_recall,
                'comm_kb': avg_bit_loss,
            }
            latest_ckpt = os.path.join('models_temp', f'{save_prefix}_latest.pth')
            best_ckpt = os.path.join('models_temp', f'{save_prefix}.pth')
            _save_checkpoint(latest_ckpt, model, optimizer, epoch, args, ckpt_metrics)
            if moda >= best_moda:
                best_moda = moda
                _save_checkpoint(best_ckpt, model, optimizer, epoch, args, ckpt_metrics)
                print(f'Saved best checkpoint: {best_ckpt}')

        print(f"maximum_MODA is {max_MODA:.2f}%, minimum_bits_loss {minimum_bits_loss:.2f} KB")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='Training epoch (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    parser.add_argument('--tau_1', type=int, default=0) # for fusion model
    parser.add_argument('--tau_2', type=int, default=1) # for temporal entropy module
    parser.add_argument('--dataset_path', type=str, default='./Data/Wildtrack')
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--drop_prob', type=float, default=1e-1) # for random drop frame
    parser.add_argument('--method', type=str, default='baseline', choices=['baseline', 'proposed_jscc'])
    parser.add_argument('--disable_quantization', action='store_true')
    parser.add_argument('--jscc_channel_type', type=str, default='rayleigh', choices=['awgn', 'rayleigh'])
    parser.add_argument('--jscc_latent_channels', type=int, default=-1)
    parser.add_argument('--snr_min_db', type=float, default=0.0)
    parser.add_argument('--snr_max_db', type=float, default=20.0)
    parser.add_argument('--test_snr_db', type=float, default=5.0)
    parser.add_argument('--cross_view_heads', type=int, default=4)
    parser.add_argument('--jscc_csi_gain_scale', type=float, default=0.6)
    parser.add_argument('--jscc_importance_gain_scale', type=float, default=1.0)
    parser.add_argument('--jscc_low_snr_disable_csi_threshold', type=float, default=0.0)
    parser.add_argument('--ablate_no_jscc', action='store_true')
    parser.add_argument('--ablate_no_csi', action='store_true')
    parser.add_argument('--ablate_no_analog_channel', action='store_true')
    parser.add_argument('--ablate_no_cross_view', action='store_true')
    parser.add_argument('--snr_sweep', type=str, default='')
    parser.add_argument('--snr_sweep_resume', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--save_prefix', type=str, default='')
    parser.add_argument('--exp_name', type=str, default='')



    args = parser.parse_args()
    #test_the_code(args)
    main(args)
    print('Settings:')
    print(vars(args))
