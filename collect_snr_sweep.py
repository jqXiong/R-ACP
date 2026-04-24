import argparse
import csv
from pathlib import Path


def read_snr_csv(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'snr_db': float(r['snr_db']),
                'test_loss': float(r.get('test_loss', 'nan')),
                'test_prec': float(r.get('test_prec', 'nan')),
                'moda': float(r.get('moda', 'nan')),
                'modp': float(r.get('modp', 'nan')),
                'eval_precision': float(r.get('eval_precision', 'nan')),
                'eval_recall': float(r.get('eval_recall', 'nan')),
                'comm_kb': float(r.get('comm_kb', 'nan')),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Collect SNR sweep CSV from multiple runs into one table')
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--output_csv', type=str, default='logs/snr_sweep_summary.csv')
    parser.add_argument('--tags', type=str, default='abl_1_baseline,abl_2_plus_channel,abl_3_plus_jscc,abl_4_plus_csi,abl_5_plus_cross_view,abl_6_full')
    parser.add_argument('--epoch', type=int, default=-1, help='target epoch; -1 means latest epoch file in each run')
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    tags = [t.strip() for t in args.tags.split(',') if t.strip()]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for tag in tags:
        run_dirs = sorted([p for p in logs_dir.glob(f'*_{tag}') if p.is_dir()])
        if not run_dirs:
            continue
        run_dir = run_dirs[-1]

        if args.epoch > 0:
            candidates = [run_dir / f'snr_sweep_epoch_{args.epoch}.csv']
        else:
            candidates = sorted(run_dir.glob('snr_sweep_epoch_*.csv'))
        if not candidates:
            continue
        csv_path = candidates[-1]
        epoch_str = csv_path.stem.replace('snr_sweep_epoch_', '')

        for row in read_snr_csv(csv_path):
            summary_rows.append({
                'variant': tag,
                'run_dir': str(run_dir),
                'epoch': int(epoch_str),
                'snr_db': row['snr_db'],
                'test_loss': row['test_loss'],
                'test_prec': row['test_prec'],
                'moda': row['moda'],
                'modp': row['modp'],
                'eval_precision': row['eval_precision'],
                'eval_recall': row['eval_recall'],
                'comm_kb': row['comm_kb'],
            })

    headers = ['variant', 'run_dir', 'epoch', 'snr_db', 'test_loss', 'test_prec', 'moda', 'modp', 'eval_precision', 'eval_recall', 'comm_kb']
    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print(f'Wrote {len(summary_rows)} rows to {output_path}')


if __name__ == '__main__':
    main()
