from argparse import ArgumentParser

from pathlib import Path
import random
from fractions import Fraction

dataset_infos = {
    'potsdam':
    dict(
        root='data/potsdam',
        img_dir=dict(train='img_dir/train', val='img_dir/val'),
        ann_dir=dict(train='ann_dir/train', val='ann_dir/val'),
        img_suffix='.png',
        seg_map_suffix='.png',
    ),
    'loveda':
    dict(
        root='data/loveDA',
        img_dir=dict(train='img_dir/train', val='img_dir/val'),
        ann_dir=dict(train='ann_dir/train', val='ann_dir/val'),
        img_suffix='.png',
        seg_map_suffix='.png',
    ),
    'isaid':
    dict(
        root='data/iSAID',
        img_dir=dict(train='img_dir/train', val='img_dir/val'),
        ann_dir=dict(train='ann_dir/train', val='ann_dir/val'),
        img_suffix='.png',
        seg_map_suffix='.png',
    )
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dataset_name',
                        type=str,
                        default='potsdam',
                        choices=['potsdam', 'loveda', 'isaid'])
    parser.add_argument('ratio', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    random.seed(args.seed)

    data_infos = dataset_infos[args.dataset_name]
    print(data_infos)
    data_root = Path(data_infos['root'])
    img_suffix = data_infos['img_suffix']
    seg_map_suffix = data_infos['seg_map_suffix']
    train_img_dir = data_root / data_infos['img_dir']['train']
    train_ann_dir = data_root / data_infos['ann_dir']['train']
    train_img_files = sorted(train_img_dir.glob(f'*{img_suffix}'))

    num_files = len(train_img_files)
    file_indices = list(range(num_files))
    random.shuffle(file_indices)

    ratio = Fraction(args.ratio)

    target_dir = Path(
        f'splits/{args.dataset_name}/{ratio.numerator}_{ratio.denominator}')
    target_dir.mkdir(parents=True, exist_ok=True)
    labeled_txt = open(target_dir / 'labeled.txt', 'w')
    unlabeled_txt = open(target_dir / 'unlabeled.txt', 'w')

    labeled = []
    unlabeled = []
    for i, idx in enumerate(file_indices):
        img_file = train_img_files[idx]
        img_path = img_file.relative_to(data_root)
        ann_path = (train_ann_dir /
                    (img_file.stem + seg_map_suffix)).relative_to(data_root)

        record = f'{str(img_path)} {str(ann_path)}\n'
        if i < num_files * args.ratio:
            labeled_txt.write(record)
        else:
            unlabeled_txt.write(record)

    labeled_txt.close()
    unlabeled_txt.close()

    # val
    val_img_dir = data_root / data_infos['img_dir']['val']
    val_ann_dir = data_root / data_infos['ann_dir']['val']
    val_txt = open(target_dir.parent / 'val.txt', 'w')
    for img_file in val_img_dir.glob(f'*{img_suffix}'):
        img_path = img_file.relative_to(data_root)
        ann_path = (val_ann_dir /
                    (img_file.stem + seg_map_suffix)).relative_to(data_root)
        val_txt.write(f'{str(img_path)} {str(ann_path)}\n')

    val_txt.close()


if __name__ == '__main__':
    main()
