module load python3 cuda
export LD_LIBRARY_PATH=/mnt/sw/nix/store/24ib7yiwhzcwiry2axn4n92wq2k9k6bj-cudnn-8.9.1.23-11.8/lib:$LD_LIBRARY_PATH
. ~/Environment/GPT/bin/activate
python3 /mnt/home/wwong/MLProject/ModelLibrary/jax/distributed/check_slurm.py