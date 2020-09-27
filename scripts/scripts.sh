for num_tasks in 8 16 24 32 48 64
do
    echo $num_tasks
    python train_range.py \
    --num_procs=2       \
    --num_tasks=$num_tasks      \
    --range_l=0.10      \
    --range_r=1.10

    python train_range.py \
    --num_procs=2       \
    --num_tasks=$num_tasks      \
    --range_l=1.10      \
    --range_r=1.50

    python train_range.py \
    --num_procs=2       \
    --num_tasks=$num_tasks      \
    --range_l=1.50      \
    --range_r=1.90

    python train_range.py \
    --num_procs=2       \
    --num_tasks=$num_tasks      \
    --range_l=1.00      \
    --range_r=1.90
done