#!/bin/bash
#
#SBATCH --job-name=constiemfairadvbpr
#SBATCH --mail-type=END
#SBATCH --mail-user=armielle.noulapeu@unamur.be
#SBATCH --account=cfadvbpr
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=32768
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00

# ------------------------- work -------------------------

# Starting the job
echo "Job start at $(date)"
python user_gender_fairAdvBPR-version1-Constraint-new.py
echo "Job end at $(date)"
