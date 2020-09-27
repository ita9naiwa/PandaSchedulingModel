num_procs=2
num_tasks=$1
deadline=$2

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=0.10      \
--range_r=0.40      \
--use_deadline=$deadline

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=0.50      \
--range_r=0.90      \
--use_deadline=$deadline

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=1.00      \
--range_r=0.40      \
--use_deadline=$deadline

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=1.50      \
--range_r=1.90      \
--use_deadline=$deadline
