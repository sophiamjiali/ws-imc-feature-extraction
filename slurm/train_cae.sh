#!/bin/bash
#SBATCH --job-name=train_cae
#SBATCH --output=/slurm_logs/train_cae_%j.out
#SBATCH --error=/slurm_logs/train_cae_%j.err
#SBATCH --time=09:00:00

# activate the virtual environment
source venv/bin/activate

# run the training code
python main.py
