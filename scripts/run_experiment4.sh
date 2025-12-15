#!/bin/bash
#SBATCH --job-name=llm_pca
#SBATCH --output=logs/llm_pca_%j.out
#SBATCH --error=logs/llm_pca_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Use SLURM_SUBMIT_DIR (directory where sbatch was called from)
# If not set (running locally), fall back to script directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"
fi
PROJECT_DIR="$(pwd)"

echo "=================================="
echo "Running PCA on LLM experiment"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Project Dir: $PROJECT_DIR"
echo "=================================="

# Activate virtual environment
source /cephfs/store/gr-mc2473/eszt2/venv/bin/activate

# Define parameter ranges
PCA_COMPS_I=(16)
PCA_COMPS_O=(64)
MAX_CHARS=10000

# Total number of combinations
TOTAL_RUNS=$((${#PCA_COMPS_I[@]} * ${#PCA_COMPS_O[@]}))
CURRENT_RUN=0

echo "Running $TOTAL_RUNS combinations of parameters"
echo "PCA_COMPS_I: ${PCA_COMPS_I[@]}"
echo "PCA_COMPS_O: ${PCA_COMPS_O[@]}"
echo "MAX_CHARS: $MAX_CHARS"
echo "=================================="
echo ""

# Loop through all combinations
for pca_i in "${PCA_COMPS_I[@]}"; do
    for pca_o in "${PCA_COMPS_O[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        echo "=================================="
        echo "Run $CURRENT_RUN/$TOTAL_RUNS"
        echo "PCA_COMPS_I=$pca_i, PCA_COMPS_O=$pca_o, MAX_CHARS=$MAX_CHARS"
        echo "=================================="

        # Run the experiment
        python3 experiment4.py $pca_i $pca_o --max-chars $MAX_CHARS

        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "ERROR: Experiment failed with exit code $EXIT_CODE"
            echo "Continuing to next combination..."
        else
            echo "✓ Experiment completed successfully"
        fi

        echo ""
    done
done

echo "=================================="
echo "All experiments completed!"
echo "=================================="
echo "Generated results in: experiment4_testing/"
find experiment4_testing -type f -name "experimental_results.json" -exec ls -lh {} \; 2>/dev/null || echo "No result files found"
