import argparse
import csv
from collections import defaultdict
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


def is_dominated(a, b):
    return (b['moda'] >= a['moda'] and b['comm_kb'] <= a['comm_kb']) and (
        b['moda'] > a['moda'] or b['comm_kb'] < a['comm_kb']
    )


def pareto_front(points):
    front = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if is_dominated(p, q):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda x: (x['snr_db'], -x['moda']))
    return front


def summarize_variant(rows):
    if not rows:
        return {}
    moda_avg = sum(r['moda'] for r in rows) / len(rows)
    comm_avg = sum(r['comm_kb'] for r in rows) / len(rows)
    moda_max = max(r['moda'] for r in rows)
    comm_min = min(r['comm_kb'] for r in rows)
    return {
        'avg_moda': moda_avg,
        'avg_comm_kb': comm_avg,
        'max_moda': moda_max,
        'min_comm_kb': comm_min,
    }


def main():
    parser = argparse.ArgumentParser(description='Collect SNR sweep CSV from multiple runs into one table + Pareto exports')
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--output_csv', type=str, default='logs/snr_sweep_summary.csv')
    parser.add_argument('--pareto_csv', type=str, default='logs/snr_sweep_pareto.csv')
    parser.add_argument('--summary_csv', type=str, default='logs/snr_sweep_variant_summary.csv')
    parser.add_argument('--tags', type=str, default='abl_1_baseline,abl_2_refined_baseline,abl_3_refined_prune,abl_4_refined_prune_masked,abl_5_refined_temporal,abl_6_refined_adaptive')
    parser.add_argument('--epoch', type=int, default=-1, help='target epoch; -1 means latest epoch file in each run')
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    tags = [t.strip() for t in args.tags.split(',') if t.strip()]

    output_path = Path(args.output_csv)
    pareto_path = Path(args.pareto_csv)
    summary_path = Path(args.summary_csv)
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

    group_by_snr = defaultdict(list)
    for r in summary_rows:
        group_by_snr[r['snr_db']].append(r)

    pareto_rows = []
    for snr_db, points in sorted(group_by_snr.items(), key=lambda x: x[0]):
        front = pareto_front(points)
        for p in front:
            row = dict(p)
            row['pareto'] = 1
            pareto_rows.append(row)

    pareto_headers = headers + ['pareto']
    with pareto_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=pareto_headers)
        writer.writeheader()
        for r in pareto_rows:
            writer.writerow(r)

    by_variant = defaultdict(list)
    for r in summary_rows:
        by_variant[r['variant']].append(r)

    variant_rows = []
    for variant, rows in sorted(by_variant.items()):
        s = summarize_variant(rows)
        variant_rows.append({
            'variant': variant,
            'avg_moda': s.get('avg_moda', float('nan')),
            'avg_comm_kb': s.get('avg_comm_kb', float('nan')),
            'max_moda': s.get('max_moda', float('nan')),
            'min_comm_kb': s.get('min_comm_kb', float('nan')),
            'num_points': len(rows),
        })

    with summary_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['variant', 'avg_moda', 'avg_comm_kb', 'max_moda', 'min_comm_kb', 'num_points'])
        writer.writeheader()
        for r in variant_rows:
            writer.writerow(r)

    print(f'Wrote {len(summary_rows)} rows to {output_path}')
    print(f'Wrote {len(pareto_rows)} Pareto rows to {pareto_path}')
    print(f'Wrote {len(variant_rows)} variant summary rows to {summary_path}')


if __name__ == '__main__':
    main()
