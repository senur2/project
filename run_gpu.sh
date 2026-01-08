module load conda
conda activate XXX(your environement to put)
torchrun --nnodes=1 --nproc-per-node=1 demo_gpu.py
torchrun --nnodes=1 --nproc-per-node=2 demo_gpu.py
