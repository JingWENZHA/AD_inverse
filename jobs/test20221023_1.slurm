#!/bin/bash -l

#SBATCH --job-name="test20221023_1"
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --time=2-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-user=zhanj318@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output="jobs_oe/test20221023_1-%j.o"
#SBATCH --error="jobs_oe/test20221023_1-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python inverse_PINN_AD_trueY_pt.py --name test20221023 --log_path logs --mode origin --lr 0.001 --epoch 40000 --epoch_step 1000 --save_step 5000 --main_path . --seed 1 --sw 0 --sw_step 50000 --id 1
