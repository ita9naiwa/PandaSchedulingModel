num_procs=4
num_tasks=32

python train_greedy_np.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=0.10      \
--range_r=0.90      

python train_greedy_np.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=1.00      \
--range_r=1.90      \
--use_cuda=1


python train_greedy_np.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=2.00      \
--range_r=2.90      

python train_greedy_np.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=3.00      \
--range_r=3.90      
