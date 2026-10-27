#!/bin/bash
# Lab benchmark: export one GUIDE-US fold model (benchmark/README.md, step 3).
#   sbatch projects/baseline_methods/guideus/export_benchmark.sh <fold> <path/to/best.pth> <export_root> [config]

#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --job-name=guideus_benchmark_export
#SBATCH --output=logs/guideus_baseline/%x-%j.log

FOLD=${1:?fold}; CKPT=${2:?best.pth}; OUT=${3:?export root}
CONFIG=${4:-projects/baseline_methods/guideus/guideus_cfg_benchmark_nct.yaml}

export MEDSAM_CHECKPOINT_DIR=/datasets/exactvu_pca/checkpoint_store
export EXACTVU_PCA_DATA_ROOT=/datasets/exactvu_pca
export BENCHMARK_OPTIMUM_ROOT=/datasets/exactvu_pca/OPTIMUM/UA_OL_PU_annotated_needles_multiframe
export BENCHMARK_OPTIMUM_PROSTATE_MASK_ROOT=/datasets/exactvu_pca/OPTIMUM/processed/UA_annotated_needles

module load python/3.12 cuda/12.2
module load opencv/4.12.0

source /home/obed/projects/aip-medilab/obed/medproj/medAI/med_env/bin/activate
# Run from the root of a tree that has this update and the full medAI (benchmark/README.md, "Setup").
# EXTRA_PYTHONPATH: optional, e.g. the directory containing external_libs when running from this extract.
export PYTHONPATH=$PWD${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}
srun python -m projects.baseline_methods.guideus.benchmark.export \
    --config "$CONFIG" --fold "$FOLD" --checkpoint "$CKPT" --out "$OUT" --num-workers 6
