from pathlib import Path
import random


def main():
    random.seed(42)

    data_root = Path('/data1/loveDA/')
    img_dir = data_root / 'img_dir/train'
    ann_dir = data_root / 'ann_dir/train'

    img_files = sorted(img_dir.glob('*.png'))
    num_files = len(img_files)
    indices = list(range(num_files))
    random.shuffle(indices)

    target_dir = Path('partitions/loveda/1_4')
    target_dir.mkdir(parents=True, exist_ok=True)
    labeled_txt = open(target_dir / 'labeled.txt', 'w')
    unlabeled_txt = open(target_dir / 'unlabeled.txt', 'w')

    labeled = []
    unlabeled = []
    for i, idx in enumerate(indices):
        img_file = img_files[idx]
        img_path = img_file.relative_to(data_root)
        ann_path = (ann_dir / img_file.name).relative_to(data_root)

        record = f'{str(img_path)} {str(ann_path)}\n'
        if i < num_files // 4:
            labeled_txt.write(record)
        else:
            unlabeled_txt.write(record)

    labeled_txt.close()
    unlabeled_txt.close()

    # val
    img_dir = data_root / 'img_dir/val'
    ann_dir = data_root / 'ann_dir/val'
    val_txt = open(target_dir.parent / 'val.txt', 'w')
    for img_file in img_dir.glob('*.png'):
        img_path = img_file.relative_to(data_root)
        ann_path = (ann_dir / img_file.name).relative_to(data_root)

        val_txt.write(f'{str(img_path)} {str(ann_path)}\n')

    val_txt.close()


if __name__ == '__main__':
    main()
