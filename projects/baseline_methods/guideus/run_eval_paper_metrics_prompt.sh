#!/bin/bash

#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=eval_paper_metrics_guideus_prompt
#SBATCH --output=logs/guideus_baseline/%x-%A.log

export CHECKPOINT=/scratch/obed
export JOB_ID=$SLURM_JOB_ID
export MEDSAM_CHECKPOINT_DIR=/datasets/exactvu_pca/checkpoint_store
export MEDSAM_CHECKPOINT=/datasets/exactvu_pca/checkpoint_store/sam/medsam_vit_b_cpu.pth
export WANDB_API_KEY=${WANDB_API_KEY:?set WANDB_API_KEY in your environment}
export NCT_RAW_DATA_DIR=/datasets/exactvu_pca/nct2013
export NCT_METADATA_PATH=/datasets/exactvu_pca/nct2013/metadata.csv
export EXACTVU_PCA_DATA_ROOT=/datasets/exactvu_pca
export DINOV3_LIBRARY_PATH=/home/obed/projects/aip-medilab/obed/medproj/dinov3
export DINOV3_CHECKPOINTS_PATH=/datasets/exactvu_pca/checkpoint_store/dinov3

module load python/3.12 cuda/12.2
module load opencv/4.12.0

source /home/obed/projects/aip-medilab/obed/medproj/medAI/med_env/bin/activate
srun python -m projects.baseline_methods.guideus.eval_paper_metrics \
  --cfg projects/baseline_methods/guideus/guideus_cfg0_optimum_prompt.yaml \
  --ckpts \
    fold0=/scratch/obed/guideus_baseline/guideus_optimum_prompt_0/6226101/best.pth \
    fold1=/scratch/obed/guideus_baseline/guideus_optimum_prompt_1/6226102/best.pth \
    fold2=/scratch/obed/guideus_baseline/guideus_optimum_prompt_2/6226103/best.pth \
    fold3=/scratch/obed/guideus_baseline/guideus_optimum_prompt_3/6226104/best.pth \
    fold4=/scratch/obed/guideus_baseline/guideus_optimum_prompt_4/6226105/best.pth \
  --out projects/baseline_methods/guideus/results_paper_metrics_prompt.json
