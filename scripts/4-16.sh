num_procs=4
num_tasks=16

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=0.10      \
--range_r=2.00      

python train_greedy.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=2.00      \
--range_r=3.90      
