num_procs=4
num_tasks=32

python train_range.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=0.10      \
--range_r=1.00      \
--use_deadline=True

python train_range.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=1.00      \
--range_r=1.50      \
--use_deadline=True

python train_range.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=1.50      \
--range_r=2.00      \
--use_deadline=True

python train_range.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=1.00      \
--range_r=2.00      \
--use_deadline=True

python train_range.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=2.00      \
--range_r=2.50      \
--use_deadline=True

python train_range.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=2.50      \
--range_r=3.00      \
--use_deadline=True

python train_range.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=3.00      \
--range_r=3.50      \
--use_deadline=True

python train_range.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=3.00      \
--range_r=4.00      \
--use_deadline=True