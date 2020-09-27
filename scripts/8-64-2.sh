num_procs=8
num_tasks=64

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=0.10      \
--range_r=0.90      \
--use_deadline=True

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=1.00      \
--range_r=1.90      \
--use_deadline=True

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=2.00      \
--range_r=2.90      \
--use_deadline=True


python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=3.00      \
--range_r=3.90      \
--use_deadline=True

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=4.00      \
--range_r=4.90      \
--use_deadline=True
