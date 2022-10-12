#!/bin/bash -l

#SBATCH --job-name="test20221004_9"
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --time=2-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-user=zhanj318@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output="jobs_oe/test20221004_9-%j.o"
#SBATCH --error="jobs_oe/test20221004_9-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python inverse_PINN_AD.py --log_path logs/test20221004_9.txt --mode origin --lr 0.001 --epoch 200 --epoch_step 10 --save_step 10 --main_path . --seed 9 --sw 0 --sw_step 50000
