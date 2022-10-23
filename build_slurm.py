import argparse


draft = """#!/bin/bash -l

#SBATCH --job-name="{0}_{9}"
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --time=2-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-user=zhanj318@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output="jobs_oe/{0}_{9}-%j.o"
#SBATCH --error="jobs_oe/{0}_{9}-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python {1} --name {0} --log_path {2} --mode {3} --lr {4} --epoch {5} --epoch_step {6} --save_step {7} --main_path {8} --seed {9} --sw {10} --sw_step {11} --id {9}
"""


def build_slurm(args):
    if "-" in args.id:
        start_index = int(args.id.split("-")[0])
        last_index = int(args.id.split("-")[1])
        id_range = range(start_index, last_index + 1)
    else:
        id_range = [int(args.id)]
    files_string = "{}/{}_{}.slurm".format(args.slurm_path, args.name, args.id)
    for one_id in id_range:
        with open("{}/{}_{}.slurm".format(args.slurm_path, args.name, one_id), "w") as f:
            f.write(draft.format(
                args.name,
                args.python,
                args.log_path,
                args.mode,
                args.lr,
                args.epoch,
                args.epoch_step,
                args.save_step,
                args.main_path,
                one_id,
                args.sw,
                args.sw_step
            ))
    print("build slurm file(s) \"{}\" successfully!".format(files_string))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slurm_path", type=str, default="jobs/", help="slurm_path")
    parser.add_argument("--id", type=str, default="1-5", help="id")
    parser.add_argument("--name", type=str, default="test", help="name")
    parser.add_argument("--python", type=str, default="ModelBYCC.py", help="python file name")
    parser.add_argument("--epoch", type=int, default=100, help="epoch")
    parser.add_argument("--log_path", type=str, default="logs", help="log path")
    parser.add_argument("--mode", type=str, default="origin", help="continue or origin")
    parser.add_argument("--epoch_step", type=int, default=10, help="epoch_step")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
    parser.add_argument("--main_path", default=".", help="main_path")
    parser.add_argument("--save_step", type=int, default=100, help="save_step")
    parser.add_argument("--sw", type=int, default=0, help="sliding window flag")
    parser.add_argument("--sw_step", type=int, default=50000, help="sliding window step")
    # parser.add_argument("--seed", type=int, default=0, help="seed")
    opt = parser.parse_args()
    build_slurm(opt)
