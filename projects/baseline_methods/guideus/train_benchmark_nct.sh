#!/bin/bash
# Lab benchmark: GUIDE-US on NCT2013 only, one array task per fold (benchmark/README.md, step 1).
#   sbatch --array=0-4 projects/baseline_methods/guideus/train_benchmark_nct.sh

#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --job-name=guideus_benchmark_nct
#SBATCH --output=logs/guideus_baseline/%x-%A-%a.log

export CHECKPOINT=/scratch/obed
export JOB_ID=$SLURM_JOB_ID
export MEDSAM_CHECKPOINT_DIR=/datasets/exactvu_pca/checkpoint_store
export MEDSAM_CHECKPOINT=/datasets/exactvu_pca/checkpoint_store/sam/medsam_vit_b_cpu.pth
export WANDB_API_KEY=${WANDB_API_KEY:?set WANDB_API_KEY in your environment}
export NCT_RAW_DATA_DIR=/datasets/exactvu_pca/nct2013
export NCT_METADATA_PATH=/datasets/exactvu_pca/nct2013/metadata.csv
export EXACTVU_PCA_DATA_ROOT=/datasets/exactvu_pca

module load python/3.12 cuda/12.2
module load opencv/4.12.0

source /home/obed/projects/aip-medilab/obed/medproj/medAI/med_env/bin/activate
# Run from the root of a tree that has this update and the full medAI (benchmark/README.md, "Setup").
# EXTRA_PYTHONPATH: optional, e.g. the directory containing external_libs when running from this extract.
export PYTHONPATH=$PWD${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}
srun python -m projects.baseline_methods.guideus.guideus_pnf_train \
    -c projects/baseline_methods/guideus/guideus_cfg_benchmark_nct.yaml fold=${SLURM_ARRAY_TASK_ID:?use --array=0-4}
