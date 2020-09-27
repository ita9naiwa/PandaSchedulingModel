num_procs=2
num_tasks=16

python train_greedy_np.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=0.10      \
--range_r=0.90 \
--use_deadline=1


python train_greedy_np.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=1.00      \
--range_r=1.90 \
--use_deadline=1

