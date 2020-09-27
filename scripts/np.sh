num_procs=2
num_tasks=16

python train_greedy_np.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=0.10      \
--range_r=1.00      \
--use_deadline=True


python train_greedy_np.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=1.10      \
--range_r=1.90      \
--use_deadline=True