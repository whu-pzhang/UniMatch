#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# loveda
config=configs/loveda.yaml
# labeled_id_path=partitions/loveda/1_4/labeled.txt
labeled_id_path=partitions/loveda/train.txt
unlabeled_id_path=partitions/loveda/1_4/unlabeled.txt
save_path=exp/loveda/fullsup

# deepglobe road


mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    supervised.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt