num_procs=8
num_tasks=64

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=5.00      \
--range_r=5.90      \
--use_deadline=True


python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=6.00      \
--range_r=6.90      \
--use_deadline=True

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=7.00      \
--range_r=7.90      \
--use_deadline=True
