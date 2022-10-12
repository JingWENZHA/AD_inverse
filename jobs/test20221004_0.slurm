#!/bin/bash -l

#SBATCH --job-name="test20221004_0"
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --time=2-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-user=zhanj318@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output="jobs_oe/test20221004_0-%j.o"
#SBATCH --error="jobs_oe/test20221004_0-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python inverse_PINN_AD.py --log_path logs/test20221004_0.txt --mode origin --lr 0.001 --epoch 20000 --epoch_step 2000 --save_step 5000 --main_path . --seed 0 --sw 0 --sw_step 50000
