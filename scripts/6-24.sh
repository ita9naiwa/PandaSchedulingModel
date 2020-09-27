num_procs=6
num_tasks=24

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=3.00      \
--range_r=4.40      \
--use_deadline=1

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=0.10      \
--range_r=1.40      \
--use_deadline=1
python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=1.50      \
--range_r=2.90      \
--use_deadline=1

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=4.50      \
--range_r=5.90      \
--use_deadline=1
