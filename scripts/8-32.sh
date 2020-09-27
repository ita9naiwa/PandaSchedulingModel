num_procs=8
num_tasks=32

python train_greedy_np.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=0.10      \
--range_r=1.90      
#--use_deadline=True

python train_greedy_np.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=2.00      \
--range_r=3.90      
#--use_deadline=True

python train_greedy_np.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=4.00      \
--range_r=5.90      #;--use_deadline=True

python train_greedy_np.py \
--num_procs=$num_procs       \
--num_tasks=$num_tasks      \
--range_l=6.00      \
--range_r=7.90      
#--use_deadline=True
