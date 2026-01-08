#!/bin/bash

MODEL="mnasnet1_0"
DATASET="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
OUTPUT="resultats.txt"
# --- Double Boucle ---

# 1. On fait varier le nombre de cœurs de 1 à 8
for CORES in {1..8}
do
    # 2. On fait varier le Batch Size
    for BATCH in 16 32 64 128
    do
        echo "--------------------------------------------------"
        echo "TESTING: $CORES Cores | Batch Size: $BATCH"
        echo "--------------------------------------------------"

        # Lancement de torchrun
        # --nproc_per_node=$CORES : C'est ici qu'on définit le nombre de cœurs utilisés
        torchrun --nnodes=1 --nproc_per_node=$CORES demo_cpu.py \
            "$MODEL" \
            "$DATASET" \
            "$BATCH" \
            "$OUTPUT" \
            "$CORES"

        # Petite pause pour ne pas surchauffer ou saturer les I/O
        sleep 1
    done
done

echo "=================================================="
echo "Benchmark terminé. Résultats dans $OUTPUT"
echo "=================================================="